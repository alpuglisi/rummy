import copy
import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from env.vectorized_env import VectorizedRummyEnv
from models.ppo_network import AUX_GROUPS, RummyActorCritic
from config import PPOConfig
import glob

from env.vectorized_env import AUX_DIM, KNOWN_CARDS, OBS_DIM, UNSEEN_CARDS, adapt_obs, blank_known
from evaluate import DeckOnlyPolicy, GreedyPolicy, ModelPolicy, RandomPolicy, play_matches
from search import SearchPolicy

class RolloutBuffer:
    def __init__(self, cfg: PPOConfig, device: torch.device):
        self.states = torch.zeros((cfg.num_steps, cfg.num_envs, cfg.obs_dim), dtype=torch.float32).to(device)
        self.masks = torch.zeros((cfg.num_steps, cfg.num_envs, cfg.action_dim), dtype=torch.bool).to(device)
        self.actions = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.long).to(device)
        self.logprobs = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.float32).to(device)
        self.values = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.float32).to(device)
        self.dones = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.bool).to(device)
        # False where a pool opponent chose the action: excluded from the policy loss.
        self.valid = torch.ones((cfg.num_steps, cfg.num_envs), dtype=torch.bool).to(device)
        # Auxiliary targets the engine knows at the stored state, and labels
        # filled in later from what happened next (-1 = unknown, masked out).
        self.aux = torch.zeros((cfg.num_steps, cfg.num_envs, AUX_DIM), dtype=torch.float32).to(device)
        self.players = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.long).to(device)
        self.next_discard = torch.full((cfg.num_steps, cfg.num_envs), -1, dtype=torch.long).to(device)
        self.turns_left = torch.full((cfg.num_steps, cfg.num_envs), -1.0, dtype=torch.float32).to(device)
        self.goes_out = torch.full((cfg.num_steps, cfg.num_envs), -1, dtype=torch.long).to(device)
        # Which auxiliary target groups are active for the game each state belongs to.
        self.aux_mask = torch.ones((cfg.num_steps, cfg.num_envs, len(AUX_GROUPS)), dtype=torch.float32).to(device)
        self.step = 0
        self.device = device

    def store(self, state, mask, action, logprob, reward, value, done, valid=None, aux=None, players=None,
              aux_mask=None):
        if valid is not None:
            self.valid[self.step] = valid.to(self.device)
        if aux is not None:
            self.aux[self.step] = aux.to(self.device)
        if aux_mask is not None:
            self.aux_mask[self.step] = aux_mask
        if players is not None:
            self.players[self.step] = players.to(self.device)
        self.states[self.step] = state.to(self.device)
        self.masks[self.step] = mask.to(self.device)
        self.actions[self.step] = action.to(self.device)
        self.logprobs[self.step] = logprob.to(self.device)
        self.rewards[self.step] = reward.to(self.device)
        self.values[self.step] = value.squeeze(-1).to(self.device)
        self.dones[self.step] = done.to(self.device)
        self.step += 1

    def compute_advantages(self, next_value, next_state, next_done, gamma=0.99, gae_lambda=0.95):
        advantages = torch.zeros_like(self.rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.step)):
            if t == self.step - 1:
                nextnonterminal = 1.0 - next_done.float()
                nextvalues = next_value
                next_is_same_player = (next_state[..., -3] == 1.0).float()
            else:
                nextnonterminal = 1.0 - self.dones[t + 1].float()
                nextvalues = self.values[t + 1]
                next_is_same_player = (self.states[t + 1][..., -3] == 1.0).float()
                
            perspective_coef = 2.0 * next_is_same_player - 1.0
            
            delta = self.rewards[t] + gamma * nextvalues * perspective_coef * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * perspective_coef * nextnonterminal * lastgaelam
            
        returns = advantages + self.values
        return advantages, returns

    def clear(self):
        self.step = 0
        self.next_discard.fill_(-1)
        self.turns_left.fill_(-1.0)
        self.goes_out.fill_(-1)

class PPOTrainer:
    def __init__(self, config: PPOConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        # TF32 matmuls: free speed on Ampere+ for these linear layers at no cost to PPO.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        if config.obs_dim != OBS_DIM:
            raise ValueError(f"config.obs_dim is {config.obs_dim} but the engine observation is {OBS_DIM} wide")
        self.envs = VectorizedRummyEnv(config.num_envs, config.env_threads,
                                       blank_known_prob=config.blank_known_prob, hand_size=config.hand_size)
        self.model = RummyActorCritic(config.obs_dim, config.action_dim, config.hidden_size,
                                      config.num_layers, config.residual).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.buffer = RolloutBuffer(config, self.device)

        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.frozen_model = None
        self.best_model = None      # incumbent best policy (see config.best_checkpoint)
        self.best_step = 0
        self.evals_run = 0
        self.rng = np.random.default_rng()
        # Separate generator for the per-game auxiliary dropout, seeded from OS
        # entropy so every game's choice is independent of everything else.
        self.aux_rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
        self.aux_mask_host = np.ones((config.num_envs, len(AUX_GROUPS)), dtype=np.float32)
        self.aux_mask = torch.ones((config.num_envs, len(AUX_GROUPS)), dtype=torch.float32, device=self.device)
        self.last_aux = None
        self.teacher = SearchPolicy(self.model, self.device, worlds=config.distill_worlds,
                                    max_actions=config.distill_actions,
                                    seed=int(self.rng.integers(0, 2**31)),
                                    horizon=config.distill_horizon, endgame=config.search_endgame,
                                    reward_scale=config.reward_scale)

        # Opponent pool: frozen policies; each game is either pure self-play
        # (opponent -1) or has one pool member controlling one seat.
        self.pool = []
        if config.pool_fraction > 0 and config.pool_init_dir and os.path.isdir(config.pool_init_dir):
            for path in sorted(glob.glob(os.path.join(config.pool_init_dir, "*.pth"))):
                model = RummyActorCritic.from_state_dict(torch.load(path, map_location=self.device))
                self.pool.append(model.to(self.device).eval())
            self.pool = self.pool[-config.pool_size:]
            if self.pool:
                print(f"Opponent pool seeded with {len(self.pool)} checkpoint(s) from {config.pool_init_dir}/")
        self.pool_wins = [0.0] * len(self.pool)
        self.pool_games = [0.0] * len(self.pool)
        # pool_wins / pool_games: decayed win and game counts of the learner
        # against each member, for PFSP sampling (kept aligned with self.pool).
        # Per-game bookkeeping (opponent, seat, round start, dropout draws)
        # lives on the host: it is tiny, and keeping it there means no step
        # has to wait for the GPU to read it back. Device copies are kept for
        # the tensors the losses need.
        self.opponent_host = np.full(config.num_envs, -1, dtype=np.int64)
        self.opponent_seat_host = np.ones(config.num_envs, dtype=np.int32)
        self.opponent = torch.full((config.num_envs,), -1, dtype=torch.long, device=self.device)
        self.opponent_seat = torch.ones(config.num_envs, dtype=torch.int32, device=self.device)
        self.round_start = np.zeros(config.num_envs, dtype=np.int64)

    def resample_aux_mask(self, which):
        """Draw a fresh set of active auxiliary target groups for the games in `which`
        (a host bool array; each group is dropped for the whole game with
        probability aux_dropout)."""
        envs = np.flatnonzero(which)
        if len(envs) == 0 or self.cfg.aux_dropout <= 0:
            return
        keep = self.aux_rng.random((len(envs), len(AUX_GROUPS))) >= self.cfg.aux_dropout
        self.aux_mask_host[envs] = keep
        self.aux_mask.copy_(torch.from_numpy(self.aux_mask_host))

    def pool_win_rates(self):
        """Learner's decayed win rate against each pool member (0.5 until it has played them)."""
        return [w / g if g >= 1.0 else 0.5 for w, g in zip(self.pool_wins, self.pool_games)]

    def assign_opponents(self, which):
        """Pick a pool opponent (or none) and a seat for the games in `which` (a host bool array)."""
        envs = np.flatnonzero(which)
        n = len(envs)
        if n == 0:
            return
        if not self.pool or self.cfg.pool_fraction <= 0:
            self.opponent_host[envs] = -1
        else:
            use_pool = self.rng.random(n) < self.cfg.pool_fraction
            if self.cfg.pool_pfsp:
                # Prioritised fictitious self-play: favour the members the learner loses to.
                rates = np.array(self.pool_win_rates())
                weights = np.clip(1.0 - rates, 0.0, None) ** self.cfg.pool_pfsp_power + 0.05
                ids = self.rng.choice(len(self.pool), size=n, p=weights / weights.sum())
            else:
                ids = self.rng.integers(len(self.pool), size=n)
            self.opponent_host[envs] = np.where(use_pool, ids, -1)
            self.opponent_seat_host[envs] = self.rng.integers(1, 3, size=n)
        self.opponent.copy_(torch.from_numpy(self.opponent_host))
        self.opponent_seat.copy_(torch.from_numpy(self.opponent_seat_host))

    def update_pool_stats(self, reward, players, done, decay=0.995):
        """Record who won each finished pool game (from the terminal reward's sign).
        All three arguments are host arrays."""
        finished = np.flatnonzero(done & (self.opponent_host >= 0))
        for e in finished:
            actor_is_pool = players[e] == self.opponent_seat_host[e]
            r = reward[e]
            won = 0.5 if r == 0 else float((r > 0) != actor_is_pool)
            pid = int(self.opponent_host[e])
            self.pool_games[pid] = self.pool_games[pid] * decay + 1.0
            self.pool_wins[pid] = self.pool_wins[pid] * decay + won

    def add_pool_member(self, model):
        self.pool.append(model)
        self.pool_wins.append(0.0)
        self.pool_games.append(0.0)
        if len(self.pool) > self.cfg.pool_size:
            self.pool.pop(0)
            self.pool_wins.pop(0)
            self.pool_games.pop(0)

    def backfill_labels(self, t, action, players, round_ends, round_outs, done):
        """Fill in labels that depend on what happens next.

        A turn is two consecutive buffer steps of the same game (draw, then
        discard), so when a player discards at step t the opponent's last
        turn was steps t-2 and t-3: those states get 'opponent's next discard'.
        When a round ends, every state since it started gets the turns that
        were left and who went out (me / opponent / nobody)."""
        buf = self.buffer
        discards = action >= 53                      # host arrays throughout; device writes use host-built indices
        for k in (2, 3):
            if t - k < 0:
                break
            envs = np.flatnonzero(discards & (self.round_start <= t - k))
            if len(envs):
                buf.next_discard[t - k, torch.from_numpy(envs).to(self.device)] = \
                    torch.from_numpy(action[envs] - 53).to(self.device)
        for e in np.flatnonzero(round_ends | done):
            i0 = int(self.round_start[e])
            steps = torch.arange(i0, t + 1, device=self.device)
            buf.turns_left[i0:t + 1, e] = (t - steps).float() / 2.0 / 40.0
            out = int(round_outs[e])
            if out == 0:
                buf.goes_out[i0:t + 1, e] = 2
            else:
                buf.goes_out[i0:t + 1, e] = torch.where(buf.players[i0:t + 1, e] == out, 0, 1)
            self.round_start[e] = t + 1

    @torch.no_grad()
    def pool_actions(self, action, state, mask, players, players_host):
        """Override the learner's actions where a pool opponent is to move; return the learner-turn mask.

        Which games each member controls is worked out from the host copies, so
        the GPU never has to be read back mid-step."""
        turn_host = (self.opponent_host >= 0) & (players_host == self.opponent_seat_host)
        for pid in np.unique(self.opponent_host[turn_host]):
            idx = torch.from_numpy(np.flatnonzero(turn_host & (self.opponent_host == pid))).to(self.device)
            model = self.pool[int(pid)]
            logits, _ = model(adapt_obs(state[idx], model.obs_dim), mask[idx])
            action[idx] = torch.distributions.Categorical(logits=logits).sample()
        return ~torch.from_numpy(turn_host).to(self.device)

    def anneal_lr(self, global_step):
        cfg = self.cfg
        progress = global_step / cfg.total_timesteps
        frac = min(1.0, max(0.0, (progress - cfg.lr_anneal_start) / max(1e-9, 1.0 - cfg.lr_anneal_start)))
        lr = cfg.learning_rate + (cfg.lr_final - cfg.learning_rate) * frac
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr
        
    def train(self):
        state, mask = self.envs.reset()
        state = state.to(self.device)
        mask = mask.to(self.device)
        done = torch.zeros(self.cfg.num_envs, dtype=torch.bool).to(self.device)
        players_host = self.envs.current_players().numpy()
        players = torch.from_numpy(players_host).to(self.device)
        aux_targets = self.envs.aux_targets()
        everyone = np.ones(self.cfg.num_envs, dtype=bool)
        self.assign_opponents(everyone)
        self.resample_aux_mask(everyone)

        global_step = 0
        update = 0
        while global_step < self.cfg.total_timesteps:
            t_start = time.time()
            lr = self.anneal_lr(global_step)
            learner_steps = torch.zeros((), device=self.device)
            self.round_start[:] = 0
            for t in range(self.cfg.num_steps):
                with torch.no_grad():
                    action, logprob, value = self.model.get_action(state, mask)
                valid = self.pool_actions(action, state, mask, players, players_host)
                learner_steps += valid.sum()

                # The one host read per step: the engine needs the actions.
                action_host = action.cpu().numpy()
                next_state, next_mask, reward, next_done, round_ends, round_outs = self.envs.step(action_host)
                reward_host = reward.numpy() * self.cfg.reward_scale
                done_host = next_done.numpy()
                reward = torch.from_numpy(reward_host).to(self.device)
                next_done = next_done.to(self.device)

                self.buffer.store(state, mask, action, logprob, reward, value, done, valid, aux_targets, players,
                                  self.aux_mask)
                self.backfill_labels(t, action_host, players_host, round_ends.numpy(), round_outs.numpy(), done_host)
                self.update_pool_stats(reward_host, players_host, done_host)

                state = next_state.to(self.device)
                mask = next_mask.to(self.device)
                done = next_done
                players_host = self.envs.current_players().numpy()
                players = torch.from_numpy(players_host).to(self.device)
                aux_targets = self.envs.aux_targets()
                if done_host.any():
                    self.assign_opponents(done_host)
                    self.resample_aux_mask(done_host)
                global_step += self.cfg.num_envs

            t_rollout = time.time()
            with torch.no_grad():
                _, next_value = self.model(state, mask)
                next_value = next_value.squeeze(-1)

            advantages, returns = self.buffer.compute_advantages(
                next_value, state, done, self.cfg.gamma, self.cfg.gae_lambda
            )

            distill = None
            if self.cfg.distill_coef > 0 and update % self.cfg.distill_every == 0:
                distill = self.collect_search_targets()
                self.writer.add_scalar("Search/TeacherAgreement", distill["agreement"], global_step)
                self.writer.add_scalar("Search/RolloutSteps", distill["rollout_steps"], global_step)
                self.writer.add_scalar("Perf/DistillSec", distill["seconds"], global_step)
                self.writer.add_scalar("Search/TeacherBelief", float(self.teacher.belief), global_step)
            t_distill = time.time()

            actor_loss, critic_loss, distill_loss, aux = self.optimize(advantages, returns, distill)
            self.last_aux = aux
            t_optimize = time.time()

            self.writer.add_scalar("Loss/Actor", actor_loss, global_step)
            self.writer.add_scalar("Loss/Critic", critic_loss, global_step)
            if distill is not None:
                self.writer.add_scalar("Loss/Distill", distill_loss, global_step)
            self.writer.add_scalar("Train/LearningRate", lr, global_step)
            if self.cfg.aux_coef > 0:
                for name, value in aux.items():
                    if name not in ("precision", "baseline"):
                        self.writer.add_scalar(f"Aux/{name}", value, global_step)
                self.writer.add_scalar("Aux/TopKPrecision", aux["precision"], global_step)
                self.writer.add_scalar("Aux/TopKBaseline", aux["baseline"], global_step)
            self.writer.add_scalar("Pool/Size", len(self.pool), global_step)
            if self.pool:
                rates = [r for r, g in zip(self.pool_win_rates(), self.pool_games) if g >= 5]
                if rates:
                    self.writer.add_scalar("Pool/HardestWinRate", min(rates), global_step)
                    self.writer.add_scalar("Pool/MeanWinRate", sum(rates) / len(rates), global_step)
            self.writer.add_scalar("Pool/LearnerStepFraction",
                                   float(learner_steps) / (self.cfg.num_steps * self.cfg.num_envs), global_step)
            self.writer.add_scalar("Reward/Average_Return", returns.mean().item(), global_step)

            self.buffer.clear()
            samples = self.cfg.num_envs * self.cfg.num_steps
            train_sps = samples / (t_optimize - t_start)
            print(f"Global Step: {global_step} / {self.cfg.total_timesteps}  ({train_sps:,.0f} steps/s)")

            update += 1
            if self.cfg.pool_fraction > 0 and update % self.cfg.pool_add_every == 0:
                self.add_pool_member(copy.deepcopy(self.model).eval())
            if update % self.cfg.eval_interval == 0:
                self.evaluate(global_step)
            t_end = time.time()

            self.writer.add_scalar("Perf/RolloutSec", t_rollout - t_start, global_step)
            self.writer.add_scalar("Perf/OptimizeSec", t_optimize - t_distill, global_step)
            self.writer.add_scalar("Perf/EvalSec", t_end - t_optimize, global_step)
            self.writer.add_scalar("Perf/TrainStepsPerSec", train_sps, global_step)
            self.writer.add_scalar("Perf/StepsPerSec", samples / (t_end - t_start), global_step)
            
            current_millions = global_step // 1_000_000

            
            previous_millions = (global_step - (self.cfg.num_envs * self.cfg.num_steps)) // 1_000_000

            
            if current_millions > previous_millions:

            
                self.save_checkpoint(f"checkpoints/ppo_rummy_{global_step}.pth")

            
                print(f"Checkpoint saved at step {global_step}!")

    def collect_search_targets(self):
        """Run the search on a sample of live training positions and turn the
        per-action outcomes into a target distribution for the policy."""
        cfg = self.cfg
        t0 = time.time()
        idx = self.rng.choice(cfg.num_envs, size=min(cfg.distill_positions, cfg.num_envs), replace=False)
        envs = [self.envs.clone(i) for i in idx]

        self.model.eval()
        # Belief-weighted worlds only once the opponent-hand head beats random guessing
        # by a margin; a weak belief makes the worlds wrong in a consistent direction.
        gain = self.last_aux["precision"] - self.last_aux["baseline"] if self.last_aux else 0.0
        self.teacher.belief = bool(cfg.distill_belief and cfg.aux_coef > 0
                                   and gain >= cfg.distill_belief_min_gain)
        self.teacher.reset_stats()
        stats = self.teacher.evaluate_actions(envs)

        # The policy learns from its own view of each position: hide the
        # opponent-known channel where that training game has it hidden.
        obs = np.stack([e.get_state() for e in envs])
        masks = np.stack([e.get_legal_actions() for e in envs]).astype(bool)
        blank = self.envs.blank[idx]
        if blank.any():
            obs[blank] = blank_known(obs[blank])

        targets = np.zeros((len(envs), cfg.action_dim), dtype=np.float32)
        for j, s in enumerate(stats):
            acts = np.fromiter(s.keys(), dtype=np.int64)
            vals = np.array([s[a][0] for a in acts])
            w = np.exp((vals - vals.max()) / cfg.distill_temperature)
            targets[j, acts] = w / w.sum()

        obs_t = torch.as_tensor(obs, device=self.device)
        mask_t = torch.as_tensor(masks, device=self.device)
        with torch.no_grad():
            model_pick = self.model(obs_t, mask_t)[0].argmax(dim=-1).cpu().numpy()
        self.model.train()

        return {
            "states": obs_t,
            "masks": mask_t,
            "targets": torch.as_tensor(targets, device=self.device),
            "agreement": float((model_pick == targets.argmax(axis=1)).mean()),
            "rollout_steps": self.teacher.stats()["mean_rollout_steps"],
            "seconds": time.time() - t0,
        }

    def aux_losses(self, out, states, targets, next_discard, turns_left, goes_out, active=None):
        """Auxiliary losses for one minibatch. Returns (weighted sum, per-target stats).

        `targets` is the engine's AUX block for each state (see rummy_env.h);
        the next three are labels backfilled from later steps (-1 = unknown);
        `active` [B, len(AUX_GROUPS)] says which target groups count for each
        sample's game (per-game dropout). Every loss is a per-sample quantity
        averaged over the samples where its group is active and its label known."""
        w = self.cfg.aux_weights
        hand = states[:, :52]
        stats = {}
        per_sample = {}

        def masked_mean(values, valid):
            return (values * valid).sum() / valid.sum().clamp(min=1.0)

        opp = targets[:, 0:52]
        per_sample["opponent"] = F.binary_cross_entropy_with_logits(out["opponent"], opp, reduction="none").mean(1)
        per_sample["layoff"] = F.binary_cross_entropy_with_logits(out["layoff"], targets[:, 52:104],
                                                                  reduction="none").mean(1)
        # Only the player's own hand cards can be discarded: average over those.
        n_hand = hand.sum(1).clamp(min=1.0)
        takeable = F.binary_cross_entropy_with_logits(out["takeable"], targets[:, 104:156], reduction="none")
        per_sample["takeable"] = (takeable * hand).sum(1) / n_hand
        value_err = (out["discard_value"] - targets[:, 156:208]) ** 2
        per_sample["discard_value"] = (value_err * hand).sum(1) / n_hand
        flags = targets[:, [208, 210, 211]]   # opponent holds a meld, can go out next turn, deck completes a meld
        per_sample["flags"] = F.binary_cross_entropy_with_logits(out["flags"], flags, reduction="none").mean(1)
        per_sample["hand_points"] = (out["hand_points"].squeeze(-1) - targets[:, 209]) ** 2
        known = {}
        if "next_discard" in out:
            per_sample["next_discard"] = F.cross_entropy(out["next_discard"], next_discard, ignore_index=-1,
                                                         reduction="none")
            known["next_discard"] = (next_discard >= 0).float()
            per_sample["turns_left"] = (out["turns_left"].squeeze(-1) - turns_left) ** 2
            known["turns_left"] = (turns_left >= 0).float()
            per_sample["goes_out"] = F.cross_entropy(out["goes_out"], goes_out, ignore_index=-1, reduction="none")
            known["goes_out"] = (goes_out >= 0).float()

        losses = {}
        for i, name in enumerate(AUX_GROUPS):
            if name not in per_sample:
                continue
            valid = known.get(name, torch.ones_like(per_sample[name]))
            if active is not None:
                valid = valid * active[:, i]
            losses[name] = masked_mean(per_sample[name], valid)
        total = sum(w.get(name, 1.0) * l for name, l in losses.items())

        with torch.no_grad():
            for name, l in losses.items():
                stats[f"Loss_{name}"] = l.detach()
            stats["TotalLoss"] = total.detach()
            # Precision of the top-k predicted cards, k = opponent hand size, against a
            # random guess among the cards that could be in their hand.
            k = opp.sum(dim=1)
            top = torch.topk(out["opponent"], 26, dim=1).indices
            rank_ok = (torch.arange(26, device=states.device)[None, :] < k[:, None]).float()
            candidates = states[:, KNOWN_CARDS].sum(dim=1) + states[:, UNSEEN_CARDS].sum(dim=1)
            stats["_hits"] = (opp.gather(1, top) * rank_ok).sum()
            stats["_total"] = k.sum()
            stats["_baseline"] = (k * k / candidates.clamp(min=1)).sum()
            if "next_discard" in out:
                m = next_discard >= 0
                stats["NextDiscardAcc"] = ((out["next_discard"].argmax(1) == next_discard) & m).sum() / m.sum().clamp(min=1)
                m = goes_out >= 0
                stats["WhoGoesOutAcc"] = ((out["goes_out"].argmax(1) == goes_out) & m).sum() / m.sum().clamp(min=1)
                m = turns_left >= 0
                stats["TurnsLeftMAE"] = ((out["turns_left"].squeeze(-1) - turns_left).abs() * m).sum() / m.sum().clamp(min=1) * 40
                stats["CanGoOutAcc"] = ((out["flags"][:, 1] > 0).float() == flags[:, 1]).float().mean()
                cards = hand.sum().clamp(min=1.0)
                stats["TakeableAcc"] = (((out["takeable"] > 0).float() == targets[:, 104:156]).float() * hand).sum() / cards
                stats["DiscardValueMAE"] = ((out["discard_value"] - targets[:, 156:208]).abs() * hand).sum() / cards * 50
        return total, stats

    def optimize(self, advantages, returns, distill=None):
        b_states = self.buffer.states.view(-1, self.cfg.obs_dim)
        b_masks = self.buffer.masks.view(-1, self.cfg.action_dim)
        b_actions = self.buffer.actions.view(-1)
        b_logprobs = self.buffer.logprobs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_valid = self.buffer.valid.view(-1).float()
        b_aux = self.buffer.aux.view(-1, AUX_DIM)
        b_next = self.buffer.next_discard.view(-1)
        b_turns = self.buffer.turns_left.view(-1)
        b_out = self.buffer.goes_out.view(-1)
        b_amask = self.buffer.aux_mask.view(-1, len(AUX_GROUPS))

        valid_adv = b_advantages[b_valid > 0]
        b_advantages = (b_advantages - valid_adv.mean()) / (valid_adv.std() + 1e-8)

        num_samples = b_states.shape[0]
        # Running statistics stay on the device; one host read at the end
        # instead of a GPU sync per minibatch.
        acc = torch.zeros(3, device=self.device)   # actor, critic, distill
        aux_acc = {}
        batches = 0
        distill_batch = min(self.cfg.batch_size, len(distill["states"])) if distill is not None else 0

        for _ in range(self.cfg.epochs):
            perm = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, self.cfg.batch_size):
                idx = perm[start:start + self.cfg.batch_size]
                mb_states = b_states[idx]
                mb_masks = b_masks[idx]
                mb_actions = b_actions[idx]
                mb_old_logprobs = b_logprobs[idx]
                mb_advantages = b_advantages[idx]
                mb_returns = b_returns[idx]
                mb_valid = b_valid[idx]
                n_valid = mb_valid.sum().clamp(min=1.0)

                logits, new_values, aux_out = self.model.forward_with_aux(mb_states, mb_masks)
                dist = torch.distributions.Categorical(logits=logits)

                new_logprobs = dist.log_prob(mb_actions)
                entropy = (dist.entropy() * mb_valid).sum() / n_valid

                logratio = new_logprobs - mb_old_logprobs
                ratio = logratio.exp()

                pg_loss1 = mb_advantages * ratio
                pg_loss2 = mb_advantages * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
                actor_loss = -(torch.min(pg_loss1, pg_loss2) * mb_valid).sum() / n_valid

                critic_loss = F.mse_loss(new_values.squeeze(-1), mb_returns)

                loss = actor_loss + (self.cfg.vf_coef * critic_loss) - (self.cfg.ent_coef * entropy)

                if self.cfg.aux_coef > 0:
                    aux_loss, stats = self.aux_losses(aux_out, mb_states, b_aux[idx], b_next[idx],
                                                      b_turns[idx], b_out[idx], b_amask[idx])
                    loss = loss + self.cfg.aux_coef * aux_loss
                    for name, value in stats.items():
                        aux_acc[name] = aux_acc.get(name, 0.0) + value

                if distill is not None:
                    d_idx = torch.randint(len(distill["states"]), (distill_batch,), device=self.device)
                    d_logits, _ = self.model(distill["states"][d_idx], distill["masks"][d_idx])
                    # Illegal actions carry -inf log-probs and zero target mass; clamp keeps 0 * -inf out.
                    d_logp = F.log_softmax(d_logits, dim=-1).clamp(min=-1e4)
                    distill_loss = -(distill["targets"][d_idx] * d_logp).sum(dim=-1).mean()
                    loss = loss + self.cfg.distill_coef * distill_loss
                    acc[2] += distill_loss.detach()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                acc[0] += actor_loss.detach()
                acc[1] += critic_loss.detach()
                batches += 1

        total_a_loss, total_c_loss, total_d_loss = acc.tolist()
        sums = {k: float(v) for k, v in aux_acc.items()}
        aux = {k: v / batches for k, v in sums.items() if not k.startswith("_")}
        aux["precision"] = sums.get("_hits", 0.0) / max(sums.get("_total", 0.0), 1.0)
        aux["baseline"] = sums.get("_baseline", 0.0) / max(sums.get("_total", 0.0), 1.0)
        return total_a_loss / batches, total_c_loss / batches, total_d_loss / batches, aux

    def evaluate(self, global_step):
        self.model.eval()
        agent = ModelPolicy(self.model, self.device)
        n = self.cfg.eval_games

        # Deck-only and the blind greedy match were dropped: the first is a
        # guaranteed win under round scoring and the second tracks the sighted
        # score within noise, so they cost a third of the eval for nothing.
        hs = self.cfg.hand_size
        vs_random = play_matches(agent, RandomPolicy(global_step), n, seed=global_step, hand_size=hs)
        vs_greedy = play_matches(agent, GreedyPolicy(global_step), n, seed=global_step + 3, hand_size=hs)
        greedy_score = vs_greedy["win_rate"] + 0.5 * vs_greedy["draw_rate"]
        self.writer.add_scalar("Eval/WinRate_vs_Random", vs_random["win_rate"], global_step)
        self.writer.add_scalar("Eval/Score_vs_Greedy", greedy_score, global_step)
        self.writer.add_scalar("Eval/PenaltyRate", vs_greedy["penalty_rate"], global_step)
        self.writer.add_scalar("Eval/DeepDrawRate", vs_greedy["deep_draw_rate"], global_step)
        self.writer.add_scalar("Eval/PileTakeRate", vs_greedy["pile_take_rate"], global_step)
        self.writer.add_scalar("Eval/MeldPoints", vs_greedy["meld_points"], global_step)
        self.writer.add_scalar("Eval/MeanTurns", vs_greedy["mean_turns"], global_step)
        line = (f"  Eval: vs random {vs_random['win_rate']:.1%} | "
                f"vs greedy {greedy_score:.1%} | pile-take {vs_greedy['pile_take_rate']:.1%} | "
                f"meld pts {vs_greedy['meld_points']:.1f} | {vs_greedy['mean_turns']:.1f} turns")

        # Win rate against a snapshot of this policy from frozen_refresh evals ago;
        # above 50% means self-play is still making progress.
        if self.frozen_model is not None:
            vs_frozen = play_matches(agent, ModelPolicy(self.frozen_model, self.device), n, seed=global_step + 2,
                                     hand_size=self.cfg.hand_size)
            score = vs_frozen["win_rate"] + 0.5 * vs_frozen["draw_rate"]
            self.writer.add_scalar("Eval/Score_vs_Frozen", score, global_step)
            line += f" | vs frozen {score:.1%}"
        if self.evals_run % self.cfg.frozen_refresh == 0:
            self.frozen_model = copy.deepcopy(self.model)
        line += self.update_best(agent, global_step, n)
        if self.cfg.search_eval_every and self.evals_run % self.cfg.search_eval_every == 0:
            line += self.evaluate_search(agent, global_step)
        self.evals_run += 1

        self.model.train()
        print(line)

    def update_best(self, agent, global_step, n):
        """Promote the live policy to best.pth when it beats the incumbent
        head-to-head by the configured margin. Head-to-head is a stronger
        signal than the greedy score once the policy is well past greedy."""
        first = self.best_model is None
        if first:
            promoted = True
            note = f" | best: first checkpoint -> {self.cfg.best_checkpoint}"
        else:
            vs_best = play_matches(agent, ModelPolicy(self.best_model, self.device), n, seed=global_step + 6,
                                   hand_size=self.cfg.hand_size)
            score = vs_best["win_rate"] + 0.5 * vs_best["draw_rate"]
            self.writer.add_scalar("Eval/Score_vs_Best", score, global_step)
            promoted = score >= 0.5 + self.cfg.best_margin
            note = f" | vs best {score:.1%} " + ("(promoted)" if promoted else f"(best is step {self.best_step:,})")
        if promoted:
            self.best_model = copy.deepcopy(self.model).eval()
            self.best_step = global_step
            self.save_checkpoint(self.cfg.best_checkpoint)
        self.writer.add_scalar("Eval/BestStep", self.best_step, global_step)
        return note

    def evaluate_search(self, plain, global_step):
        cfg = self.cfg
        search = SearchPolicy(self.model, self.device, worlds=cfg.search_worlds,
                              max_actions=cfg.search_actions, seed=global_step,
                              horizon=cfg.search_horizon, endgame=cfg.search_endgame,
                              reward_scale=cfg.reward_scale)
        t0 = time.time()
        vs_plain = play_matches(search, plain, cfg.search_eval_games, seed=global_step + 4, hand_size=cfg.hand_size)
        vs_greedy = play_matches(search, GreedyPolicy(global_step), cfg.search_eval_games, seed=global_step + 5,
                                 hand_size=cfg.hand_size)
        seconds_per_game = (time.time() - t0) / (2 * cfg.search_eval_games)
        stats = search.stats()

        plain_score = vs_plain["win_rate"] + 0.5 * vs_plain["draw_rate"]
        greedy_score = vs_greedy["win_rate"] + 0.5 * vs_greedy["draw_rate"]
        self.writer.add_scalar("Search/Score_vs_Plain", plain_score, global_step)
        self.writer.add_scalar("Search/Score_vs_Greedy", greedy_score, global_step)
        self.writer.add_scalar("Search/Agreement", stats["agreement"], global_step)
        self.writer.add_scalar("Search/ValueGap", stats["value_gap"], global_step)
        self.writer.add_scalar("Search/SecondsPerGame", seconds_per_game, global_step)
        return (f"\n  Search({cfg.search_worlds}x{cfg.search_actions}): vs plain {plain_score:.1%} | "
                f"vs greedy {greedy_score:.1%} | agrees with model {stats['agreement']:.1%} | "
                f"value gap {stats['value_gap']:+.1f} | {seconds_per_game:.2f}s/game")

    def save_checkpoint(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.model.state_dict(), path)

if __name__ == "__main__":
    cfg = PPOConfig()
    trainer = PPOTrainer(cfg)
    trainer.train()
