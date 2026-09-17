import os

import numpy as np
import torch

import rummy_engine
from env.vectorized_env import adapt_obs


class SearchPolicy:
    """Determinized Monte Carlo search on top of a trained policy.

    For each decision: sample `worlds` redeals of the hidden cards, play every
    candidate action out to the end of the current round in each world with
    the model driving both players, and pick the action with the best mean
    outcome: the searcher's score gain over the round minus the opponent's.
    Candidate actions are the model's `max_actions` most likely legal moves.
    All rollouts for all decisions are stepped together in one threaded EnvBatch.
    """

    needs_env = True

    def __init__(self, model, device, worlds=16, max_actions=4, seed=0, num_threads=0, belief=False,
                 horizon=0, endgame=True, reward_scale=0.02):
        self.model = model
        self.device = device
        self.worlds = worlds
        self.max_actions = max_actions
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
        self.reset_stats()

    @torch.no_grad()
    def _beliefs(self, obs, mask):
        obs = adapt_obs(np.asarray(obs), self.model.obs_dim)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=self.device)
        _, _, aux = self.model.forward_with_aux(obs_t, mask_t)
        return torch.sigmoid(aux["opponent"]).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _values(self, obs, mask):
        """Critic value of each position in points, from the perspective of the player to move."""
        obs = adapt_obs(np.asarray(obs), self.model.obs_dim)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=self.device)
        _, value = self.model(obs_t, mask_t)
        return value.squeeze(-1).cpu().numpy().astype(np.float64) / self.reward_scale

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

    @torch.no_grad()
    def _logits(self, obs, mask):
        obs = adapt_obs(np.asarray(obs), self.model.obs_dim)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        return self.model(obs_t, mask_t)[0]

    def _sample(self, obs, mask):
        return torch.distributions.Categorical(logits=self._logits(obs, mask)).sample().cpu().numpy()

    def evaluate_actions(self, envs):
        """Return, per env, a dict {action: (mean_outcome, n_rollouts)} for its candidates."""
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
        start = batch.scores()
        _, dones, round_ends = batch.step(sim_action)
        game_over = dones.copy()
        stopped = dones | round_ends          # rollout is finished (halted at the next observation)
        needs_value = round_ends & ~dones     # bootstrap from the new round's first position
        steps = np.ones(n, dtype=np.int64)
        boot = np.zeros(n, dtype=np.float64)

        self.rollouts += n
        self.rollout_steps += n
        while True:
            states, masks, players, idx = batch.observe_alive()
            if len(idx) == 0:
                break
            at_horizon = (self.horizon > 0) & (steps[idx] >= self.horizon) & ~stopped[idx]
            # Value the positions where rollouts stop short of the game's end.
            want = needs_value[idx] | at_horizon
            if self.endgame and want.any():
                v = self._values(states[want], masks[want])
                sign = np.where(players[want] == sim_agent[idx[want]], 1.0, -1.0)
                boot[idx[want]] = sign * v
            stop_now = stopped[idx] | at_horizon
            if stop_now.any():
                halt = np.zeros(n, dtype=bool)
                halt[idx[stop_now]] = True
                batch.halt(halt)
            live = idx[~stop_now]
            if len(live) == 0:
                break
            self.rollout_steps += len(live)
            actions = np.zeros(n, dtype=np.int64)
            actions[live] = self._sample(states[~stop_now], masks[~stop_now])
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
        outcome = own - opp + boot
        if game_over.any():
            final = batch.scores()
            mine = final[np.arange(n), sim_agent - 1]
            theirs = final[np.arange(n), 2 - sim_agent]
            outcome = np.where(game_over, own - opp + np.sign(mine - theirs) * 100.0, outcome)
        # Breaking the pile-draw obligation forfeits the game; score it as a heavy loss.
        penalised = batch.penalised()
        outcome = np.where(penalised == sim_agent, -100.0, np.where(penalised != 0, 100.0, outcome))

        results = []
        for i in range(len(envs)):
            stats = {}
            for a in candidates[i]:
                sel = (sim_env == i) & (sim_action == int(a))
                stats[int(a)] = (float(outcome[sel].mean()), int(sel.sum()))
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
