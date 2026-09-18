import os

import numpy as np
import torch

import rummy_engine
from env.vectorized_env import adapt_obs
from models.ppo_network import masked_categorical, sample_categorical

ACTIONS = 105


class SearchPolicy:
    """Determinized Monte Carlo search on top of a trained policy.

    For each decision: sample `worlds` redeals of the hidden cards, play every
    candidate action out to the end of the current round in each world with
    the model driving both players, and pick the action with the best mean
    outcome: the searcher's score gain over the round minus the opponent's.
    Candidate actions are the model's `max_actions` most likely legal moves.
    All rollouts for all decisions are stepped together in one threaded EnvBatch.

    Data path: the engine writes each observation straight into a host staging
    buffer (pinned on CUDA), which is uploaded once; every later row selection
    happens on the device, driven by small host-built index arrays, and the
    model runs over the selected rows in blocks of at most `max_rows`.
    """

    needs_env = True

    def __init__(self, model, device, worlds=16, max_actions=4, seed=0, num_threads=0, belief=False,
                 horizon=0, endgame=True, reward_scale=0.02, replies=1, max_sims=160_000, max_rows=16384):
        self.model = model
        self.device = torch.device(device)
        self.worlds = worlds
        self.max_actions = max_actions
        # Opponent replies: at the opponent's first draw after the searcher's
        # turn each rollout branches into their `replies` likeliest legal draws
        # and the candidate is scored by the worst branch (the opponent picks
        # the reply that hurts most), so the search looks one opponent decision
        # deeper than a policy-sampled rollout. 1 = no branching.
        self.replies = max(1, replies)
        self.max_sims = max_sims   # rollouts per engine batch; positions are chunked to stay under it
        self.max_rows = max(1, max_rows)   # rows per model call; larger selections are split
        # Rollouts stop at the round's end, or after `horizon` engine steps when
        # set. With `endgame`, the critic's value of the position where a rollout
        # stops (converted to points with reward_scale) is added to the score
        # margin so far, so the search values what comes after the round too.
        self.horizon = horizon
        self.endgame = endgame
        self.reward_scale = reward_scale
        self.rng = np.random.default_rng(seed)
        self.num_threads = num_threads if num_threads > 0 else (os.cpu_count() or 1)
        # Belief-weighted worlds: deal the opponent's unknown cards in proportion to the
        # auxiliary head's predicted probabilities instead of uniformly.
        if belief and not getattr(model, "has_aux", False):
            raise ValueError("belief-weighted search needs a checkpoint trained with the opponent-hand head")
        self.belief = belief
        # Two host staging sets for observations, used alternately per
        # iteration (see _observe); allocated on first use and kept across calls.
        self._staging = []
        self._staging_cap = 0
        self._staging_slot = 0
        self.reset_stats()

    # ---- staging buffers -------------------------------------------------------
    def _alloc_staging(self, cap):
        """One staging set: observation blocks as torch tensors (pinned when the
        device is CUDA) with numpy views for the engine, plus small host arrays."""
        pin = self.device.type == "cuda"
        try:
            states = torch.empty((cap, rummy_engine.OBS_SPACE_SIZE), dtype=torch.float32, pin_memory=pin)
            masks = torch.empty((cap, ACTIONS), dtype=torch.bool, pin_memory=pin)
        except RuntimeError as err:
            print(f"search: pinned staging buffers unavailable ({err}); using pageable memory")
            states = torch.empty((cap, rummy_engine.OBS_SPACE_SIZE), dtype=torch.float32)
            masks = torch.empty((cap, ACTIONS), dtype=torch.bool)
        return {
            "states": states, "masks": masks,
            "states_np": states.numpy(), "masks_np": masks.numpy(),
            "players": np.empty(cap, dtype=np.int32),
            "indices": np.empty(cap, dtype=np.int64),
            "phases": np.empty(cap, dtype=np.int8),
        }

    def _ensure_staging(self, cap):
        if cap > self._staging_cap:
            self._staging = [self._alloc_staging(cap) for _ in range(2)]
            self._staging_cap = cap

    def _observe(self, batch):
        """Observe the live games into the next staging set and upload the blocks.

        Returns (states, masks) on the device and (players, indices, phases) on
        the host, or None when no game is alive. Reuse safety: a staging set is
        overwritten only two iterations after its upload, and every iteration
        that uploads ends in a .cpu() (values, branching logits or samples) or
        in the explicit synchronisation in _evaluate_chunk, so the upload has
        completed before the set is written again.
        """
        buf = self._staging[self._staging_slot]
        self._staging_slot ^= 1
        if hasattr(batch, "observe_alive_into"):
            k = batch.observe_alive_into(buf["states_np"], buf["masks_np"], buf["players"], buf["indices"],
                                         buf["phases"])
        else:
            # Older engine builds: copy the fresh arrays into the same buffers.
            states, masks, players, indices = batch.observe_alive()
            k = len(indices)
            np.copyto(buf["states_np"][:k], states)
            np.copyto(buf["masks_np"][:k], masks)
            buf["players"][:k] = players
            buf["indices"][:k] = indices
            buf["phases"][:k] = states[:, -3] != 0.0
        if k == 0:
            return None
        states_d = buf["states"][:k].to(self.device, non_blocking=True)
        masks_d = buf["masks"][:k].to(self.device, non_blocking=True)
        # Copies of the small host arrays: the set is written again two iterations later.
        return states_d, masks_d, buf["players"][:k].copy(), buf["indices"][:k].copy(), buf["phases"][:k].copy()

    def _index(self, rows):
        return torch.as_tensor(rows, dtype=torch.int64, device=self.device)

    # ---- model calls ---------------------------------------------------------
    def _inputs(self, obs, mask):
        """Observation and mask as device tensors; numpy input is uploaded, tensors pass through."""
        obs = adapt_obs(obs if torch.is_tensor(obs) else np.asarray(obs), self.model.obs_dim)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask if torch.is_tensor(mask) else np.asarray(mask), dtype=torch.bool,
                                 device=self.device)
        return obs_t, mask_t

    def _chunked(self, fn, obs, mask):
        """fn over row blocks of at most max_rows, outputs concatenated."""
        n = obs.shape[0]
        if n <= self.max_rows:
            return fn(obs, mask)
        return torch.cat([fn(obs[i:i + self.max_rows], mask[i:i + self.max_rows])
                          for i in range(0, n, self.max_rows)])

    @torch.no_grad()
    def _beliefs(self, obs, mask):
        obs, mask = self._inputs(obs, mask)
        aux = self._chunked(lambda o, m: self.model.forward_with_aux(o, m)[2]["opponent"], obs, mask)
        return torch.sigmoid(aux).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _values(self, obs, mask):
        """Critic value of each position in points, from the perspective of the player to move."""
        obs, mask = self._inputs(obs, mask)
        value = self._chunked(lambda o, m: self.model(o, m)[1], obs, mask)
        return value.squeeze(-1).cpu().numpy().astype(np.float64) / self.reward_scale

    @torch.no_grad()
    def _logits(self, obs, mask):
        obs, mask = self._inputs(obs, mask)
        return self._chunked(lambda o, m: self.model(o, m)[0], obs, mask)

    @torch.no_grad()
    def _sample(self, obs, mask):
        obs, mask = self._inputs(obs, mask)
        draw = lambda o, m: sample_categorical(masked_categorical(self.model(o, m)[0]))
        return self._chunked(draw, obs, mask).cpu().numpy()

    def reset_stats(self):
        self.decisions = 0
        self.agreements = 0          # search picked the model's own top choice
        self.value_gap = 0.0         # mean outcome of search's pick minus the model's pick
        self.rollouts = 0
        self.rollout_steps = 0       # engine steps played across all rollouts (they run to the round's end)

    def stats(self):
        n = max(self.decisions, 1)
        return {
            "agreement": self.agreements / n,
            "value_gap": self.value_gap / n,
            "rollouts_per_decision": self.rollouts / n,
            "mean_rollout_steps": self.rollout_steps / max(self.rollouts, 1),
        }

    def evaluate_actions(self, envs):
        """Return, per env, a dict {action: (mean_outcome, n_rollouts)} for its candidates."""
        per_env = self.worlds * self.max_actions * self.replies
        chunk = max(1, self.max_sims // per_env)
        results = []
        for i in range(0, len(envs), chunk):
            results.extend(self._evaluate_chunk(envs[i:i + chunk]))
        return results

    def _top_legal(self, probs, mask, k):
        """Up to k legal actions per row, most probable first, from device tensors.

        Returns host arrays (actions [rows, k], valid [rows, k]): the valid
        entries of each row are a prefix, ordered by descending probability.
        """
        k = min(k, ACTIONS)
        scored = probs.masked_fill(~mask, -1.0)     # legal probabilities are >= 0
        top, order = torch.topk(scored, k, dim=1)
        return order.cpu().numpy(), (top >= 0.0).cpu().numpy()

    def _evaluate_chunk(self, envs):
        obs = np.stack([e.get_state() for e in envs])
        mask = np.stack([e.get_legal_actions() for e in envs]).astype(bool)
        probs = torch.softmax(self._logits(obs, mask), dim=-1).cpu().numpy()

        candidates = []
        for i in range(len(envs)):
            legal = np.flatnonzero(mask[i])
            order = legal[np.argsort(-probs[i, legal])]
            candidates.append(order[: self.max_actions])

        beliefs = self._beliefs(obs, mask) if self.belief else None

        # One rollout per (env, world, candidate). Worlds are built once per env
        # and copied per candidate so every candidate faces the same redeals.
        # The engine builds the whole batch in one call (order: env, world, candidate).
        repeats = np.array([len(c) for c in candidates], dtype=np.int32)
        seeds = np.array([[int(self.rng.integers(0, 2**31)) for _ in range(self.worlds)]
                          for _ in envs], dtype=np.uint32)
        agents = np.array([e.get_current_player() for e in envs], dtype=np.int32)
        per_env = self.worlds * repeats
        sim_env = np.repeat(np.arange(len(envs)), per_env)
        sim_agent = agents[sim_env]
        sim_action = np.concatenate([np.tile(np.asarray(c, dtype=np.int64), self.worlds) for c in candidates])

        batch = rummy_engine.EnvBatch.for_search(envs, repeats, seeds, beliefs, self.num_threads)
        n = batch.size
        self._ensure_staging(n * self.replies)   # a rollout branches at most once
        start = batch.scores()
        _, dones, round_ends = batch.step(sim_action)
        game_over = dones.copy()
        stopped = dones | round_ends          # rollout is finished (halted at the next observation)
        needs_value = round_ends & ~dones     # bootstrap from the new round's first position
        steps = np.ones(n, dtype=np.int64)
        boot = np.zeros(n, dtype=np.float64)
        group = np.arange(n)                  # (env, candidate, world) rollout each sim descends from
        branched = np.full(n, self.replies <= 1)
        reply = np.full(n, -1, dtype=np.int64)   # forced next action after branching, -1 = sample
        rank = np.zeros(n, dtype=np.int64)       # which of the opponent's replies a branch took

        self.rollouts += n
        self.rollout_steps += n
        while True:
            observed = self._observe(batch)
            if observed is None:
                break
            states, masks, players, idx, phases = observed
            forwarded = False
            at_horizon = (self.horizon > 0) & (steps[idx] >= self.horizon) & ~stopped[idx]
            # Value the positions where rollouts stop short of the game's end.
            want = needs_value[idx] | at_horizon
            if self.endgame and want.any():
                w = self._index(np.flatnonzero(want))
                v = self._values(states.index_select(0, w), masks.index_select(0, w))
                sign = np.where(players[want] == sim_agent[idx[want]], 1.0, -1.0)
                boot[idx[want]] = sign * v
                forwarded = True
            needs_value[idx[want]] = False
            stop_now = stopped[idx] | at_horizon
            if stop_now.any():
                halt = np.zeros(n, dtype=bool)
                halt[idx[stop_now]] = True
                batch.halt(halt)
            keep = np.flatnonzero(~stop_now)
            live = idx[keep]
            if len(live) == 0:
                # No model call synchronised this iteration's upload: wait for
                # it before the staging set can be written again (see _observe).
                if not forwarded and self.device.type == "cuda":
                    torch.cuda.current_stream().synchronize()
                break
            if len(keep) == len(idx):
                live_states, live_masks = states, masks
            else:
                kd = self._index(keep)
                live_states, live_masks = states.index_select(0, kd), masks.index_select(0, kd)
            live_players, live_phases = players[keep], phases[keep]

            # Branch on the opponent's first draw after the searcher's turn.
            to_branch = ~branched[live] & (live_players != sim_agent[live]) & (live_phases == 0)
            if to_branch.any():
                b = self._index(np.flatnonzero(to_branch))
                bmasks = live_masks.index_select(0, b)
                bprobs = torch.softmax(self._logits(live_states.index_select(0, b), bmasks), dim=-1)
                options, valid = self._top_legal(bprobs, bmasks, self.replies)
                rows = live[to_branch]
                counts = np.ones(n, dtype=np.int32)
                counts[rows] = valid.sum(1)
                branched[rows] = True
                offsets = np.cumsum(counts) - counts
                slots = offsets[rows][:, None] + np.arange(options.shape[1])
                reply_new = np.full(int(counts.sum()), -1, dtype=np.int64)
                reply_new[slots[valid]] = options[valid]
                # Which preference rank each new branch represents, so the
                # reduction below can average a rank over worlds before the min.
                rank_new = np.repeat(rank, counts)
                rank_new[slots[valid]] = np.broadcast_to(np.arange(options.shape[1]), options.shape)[valid]
                batch = batch.expand(counts)
                rep = lambda x: np.repeat(x, counts, axis=0)
                sim_env, sim_action, sim_agent = rep(sim_env), rep(sim_action), rep(sim_agent)
                start, game_over, stopped = rep(start), rep(game_over), rep(stopped)
                needs_value, steps, boot, group, branched = rep(needs_value), rep(steps), rep(boot), rep(group), rep(branched)
                reply, rank = reply_new, rank_new
                n = batch.size
                self.rollouts += int(counts.sum() - len(counts))
                # The expanded batch's observation is the row-repeat of the
                # live rows (halted games stay halted), so no re-observation:
                # the r-th copy of old row i sits at offsets[i] + r.
                live_counts = counts[live]
                total = int(live_counts.sum())
                reps = torch.as_tensor(live_counts, dtype=torch.int64, device=self.device)
                live_states = live_states.repeat_interleave(reps, dim=0, output_size=total)
                live_masks = live_masks.repeat_interleave(reps, dim=0, output_size=total)
                live_players = np.repeat(live_players, live_counts)
                within = np.arange(total) - np.repeat(np.cumsum(live_counts) - live_counts, live_counts)
                live = np.repeat(offsets[live], live_counts) + within

            self.rollout_steps += len(live)
            actions = np.zeros(n, dtype=np.int64)
            forced = reply[live] >= 0
            # Forced replies are applied directly; only the free rows are sampled.
            free = np.flatnonzero(~forced)
            if len(free) == len(live):
                actions[live] = self._sample(live_states, live_masks)
            else:
                actions[live[forced]] = reply[live[forced]]
                if len(free):   # else every live row was branched: _top_legal already synchronised
                    f = self._index(free)
                    actions[live[free]] = self._sample(live_states.index_select(0, f), live_masks.index_select(0, f))
            reply[live] = -1
            _, dones, round_ends = batch.step(actions)
            steps[live] += 1
            game_over[live] |= dones[live]
            stopped[live] |= (dones | round_ends)[live]
            needs_value[live] |= (round_ends & ~dones)[live]

        # Outcome: the searcher's score gain so far minus the opponent's, plus the
        # valued continuation (or +/-100 once the game is decided).
        gain = batch.scores() - start
        own = gain[np.arange(n), sim_agent - 1]
        opp = gain[np.arange(n), 2 - sim_agent]
        realized = own - opp
        if self.endgame:
            # A rollout cut at the horizon has not been paid for its round yet:
            # the engine pays the round's margin only at end_round, measured
            # from the round's start, so the critic's value where the rollout
            # stops already prices every point melded since then -- including
            # the ones melded inside the horizon window. Adding the mid-round
            # score delta on top would count those twice. Only a rollout that
            # actually reached a round or game boundary has a realized margin;
            # for the rest the bootstrap is the whole estimate. (Everything the
            # root had already banked is common to all candidates, so it
            # cancels when they are compared.)
            realized = np.where(stopped, realized, 0.0)
        outcome = realized + boot
        if game_over.any():
            final = batch.scores()
            mine = final[np.arange(n), sim_agent - 1]
            theirs = final[np.arange(n), 2 - sim_agent]
            outcome = np.where(game_over, own - opp + np.sign(mine - theirs) * 100.0, outcome)
        # Breaking the pile-draw obligation forfeits the game; score it as a heavy loss.
        penalised = batch.penalised()
        outcome = np.where(penalised == sim_agent, -100.0, np.where(penalised != 0, 100.0, outcome))

        # The opponent picks the reply that hurts the searcher most, so a
        # candidate should score as the worst of their replies. Taking that
        # minimum over the branches of a single world does not measure it: each
        # branch is one stochastic rollout, so the minimum of k of them is
        # dominated by sampling noise and sinks as k grows -- which would
        # penalise a candidate merely for leaving the opponent more legal draws,
        # something the candidate itself controls. Every candidate faces the
        # same redeals, so average a reply RANK over the worlds first and take
        # the opponent's worst rank of those averages instead.
        key = sim_env.astype(np.int64) * 256 + sim_action     # actions are < 256
        order = np.argsort(key, kind="stable")
        skey, sgroup, srank, souts = key[order], group[order], rank[order], outcome[order]
        uniq = np.unique(skey)
        lo = np.searchsorted(skey, uniq, side="left")
        hi = np.searchsorted(skey, uniq, side="right")
        seg = {int(k): (int(x), int(y)) for k, x, y in zip(uniq, lo, hi)}

        results = []
        for i in range(len(envs)):
            stats = {}
            for a in candidates[i]:
                a = int(a)
                x, y = seg[i * 256 + a]
                worlds, w_idx = np.unique(sgroup[x:y], return_inverse=True)
                by_rank = np.full((len(worlds), self.replies), np.nan)
                by_rank[w_idx, srank[x:y]] = souts[x:y]
                # A world whose rollout ended before the opponent could reply,
                # or where they had fewer legal draws than `replies`, carries
                # its last branch into the ranks it never reached.
                for r in range(1, self.replies):
                    missing = np.isnan(by_rank[:, r])
                    by_rank[missing, r] = by_rank[missing, r - 1]
                stats[a] = (float(by_rank.mean(axis=0).min()), len(worlds))
            results.append(stats)
        return results

    def act_envs(self, envs):
        stats = self.evaluate_actions(envs)
        choices = []
        for s in stats:
            model_pick = next(iter(s))               # candidates are ordered by model probability
            best = max(s, key=lambda a: s[a][0])
            choices.append(best)
            self.decisions += 1
            self.agreements += int(best == model_pick)
            self.value_gap += s[best][0] - s[model_pick][0]
        return np.array(choices, dtype=np.int64)
