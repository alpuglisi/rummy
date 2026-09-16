import copy
import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from env.vectorized_env import VectorizedRummyEnv
from models.ppo_network import RummyActorCritic
from config import PPOConfig
import glob

from env.vectorized_env import KNOWN_CARDS, UNSEEN_CARDS, adapt_obs, blank_known
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
        # Auxiliary target: the opponent's true hand at each stored state.
        self.opp_hands = torch.zeros((cfg.num_steps, cfg.num_envs, 52), dtype=torch.bool).to(device)
        self.step = 0
        self.device = device

    def store(self, state, mask, action, logprob, reward, value, done, valid=None, opp_hand=None):
        if valid is not None:
            self.valid[self.step] = valid.to(self.device)
        if opp_hand is not None:
            self.opp_hands[self.step] = opp_hand.to(self.device)
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

class PPOTrainer:
    def __init__(self, config: PPOConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        
        self.envs = VectorizedRummyEnv(config.num_envs, config.env_threads,
                                       blank_known_prob=config.blank_known_prob)
        self.model = RummyActorCritic(config.obs_dim, config.action_dim, config.hidden_size,
                                      config.num_layers, config.residual).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.buffer = RolloutBuffer(config, self.device)

        self.writer = SummaryWriter(log_dir="runs/rummy_ppo")
        self.frozen_model = None
        self.evals_run = 0
        self.rng = np.random.default_rng()
        self.teacher = SearchPolicy(self.model, self.device, worlds=config.distill_worlds,
                                    max_actions=config.distill_actions,
                                    seed=int(self.rng.integers(0, 2**31)))

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
        self.opponent = torch.full((config.num_envs,), -1, dtype=torch.long, device=self.device)
        self.opponent_seat = torch.ones(config.num_envs, dtype=torch.int32, device=self.device)

    def assign_opponents(self, which):
        """Pick a pool opponent (or none) and a seat for the games in `which`."""
        n = int(which.sum())
        if n == 0 or not self.pool or self.cfg.pool_fraction <= 0:
            self.opponent[which] = -1
            return
        use_pool = torch.rand(n, device=self.device) < self.cfg.pool_fraction
        ids = torch.randint(len(self.pool), (n,), device=self.device)
        self.opponent[which] = torch.where(use_pool, ids, torch.full_like(ids, -1))
        self.opponent_seat[which] = torch.randint(1, 3, (n,), device=self.device, dtype=torch.int32)

    @torch.no_grad()
    def pool_actions(self, action, state, mask, players):
        """Override the learner's actions where a pool opponent is to move; return the learner-turn mask."""
        pool_turn = (self.opponent >= 0) & (players == self.opponent_seat)
        if pool_turn.any():
            for pid in torch.unique(self.opponent[pool_turn]).tolist():
                idx = torch.nonzero(pool_turn & (self.opponent == pid)).squeeze(1)
                model = self.pool[pid]
                logits, _ = model(adapt_obs(state[idx], model.obs_dim), mask[idx])
                action[idx] = torch.distributions.Categorical(logits=logits).sample()
        return ~pool_turn

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
        players = self.envs.current_players().to(self.device)
        self.assign_opponents(torch.ones(self.cfg.num_envs, dtype=torch.bool, device=self.device))

        global_step = 0
        update = 0
        while global_step < self.cfg.total_timesteps:
            t_start = time.time()
            lr = self.anneal_lr(global_step)
            learner_steps = 0
            for _ in range(self.cfg.num_steps):
                with torch.no_grad():
                    action, logprob, value = self.model.get_action(state, mask)
                valid = self.pool_actions(action, state, mask, players)
                learner_steps += int(valid.sum())
                opp_hand = self.envs.opponent_hands()

                next_state, next_mask, reward, next_done = self.envs.step(action)
                reward = reward * self.cfg.reward_scale

                self.buffer.store(state, mask, action, logprob, reward, value, done, valid, opp_hand)

                state = next_state.to(self.device)
                mask = next_mask.to(self.device)
                done = next_done.to(self.device)
                players = self.envs.current_players().to(self.device)
                if done.any():
                    self.assign_opponents(done)
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
            t_distill = time.time()

            actor_loss, critic_loss, distill_loss, aux = self.optimize(advantages, returns, distill)
            t_optimize = time.time()

            self.writer.add_scalar("Loss/Actor", actor_loss, global_step)
            self.writer.add_scalar("Loss/Critic", critic_loss, global_step)
            if distill is not None:
                self.writer.add_scalar("Loss/Distill", distill_loss, global_step)
            self.writer.add_scalar("Train/LearningRate", lr, global_step)
            if self.cfg.aux_coef > 0:
                self.writer.add_scalar("Aux/OpponentHandLoss", aux["loss"], global_step)
                self.writer.add_scalar("Aux/TopKPrecision", aux["precision"], global_step)
                self.writer.add_scalar("Aux/TopKBaseline", aux["baseline"], global_step)
            self.writer.add_scalar("Pool/Size", len(self.pool), global_step)
            self.writer.add_scalar("Pool/LearnerStepFraction",
                                   learner_steps / (self.cfg.num_steps * self.cfg.num_envs), global_step)
            self.writer.add_scalar("Reward/Average_Return", returns.mean().item(), global_step)

            self.buffer.clear()
            samples = self.cfg.num_envs * self.cfg.num_steps
            train_sps = samples / (t_optimize - t_start)
            print(f"Global Step: {global_step} / {self.cfg.total_timesteps}  ({train_sps:,.0f} steps/s)")

            update += 1
            if self.cfg.pool_fraction > 0 and update % self.cfg.pool_add_every == 0:
                self.pool.append(copy.deepcopy(self.model).eval())
                self.pool = self.pool[-self.cfg.pool_size:]
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

    def optimize(self, advantages, returns, distill=None):
        b_states = self.buffer.states.view(-1, self.cfg.obs_dim)
        b_masks = self.buffer.masks.view(-1, self.cfg.action_dim)
        b_actions = self.buffer.actions.view(-1)
        b_logprobs = self.buffer.logprobs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_valid = self.buffer.valid.view(-1).float()
        b_opp = self.buffer.opp_hands.view(-1, 52).float()

        valid_adv = b_advantages[b_valid > 0]
        b_advantages = (b_advantages - valid_adv.mean()) / (valid_adv.std() + 1e-8)

        num_samples = b_states.shape[0]
        total_a_loss = 0
        total_c_loss = 0
        total_d_loss = 0
        total_aux_loss = 0
        aux_hits = aux_total = aux_baseline = 0.0
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

                logits, new_values, aux_logits = self.model.forward_with_aux(mb_states, mb_masks)
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
                    mb_opp = b_opp[idx]
                    aux_loss = F.binary_cross_entropy_with_logits(aux_logits, mb_opp)
                    loss = loss + self.cfg.aux_coef * aux_loss
                    total_aux_loss += aux_loss.item()
                    with torch.no_grad():
                        # Precision of the top-k predicted cards, k = opponent hand size, against a
                        # random guess among the cards that could be in their hand.
                        k = mb_opp.sum(dim=1)
                        top = torch.topk(aux_logits, 26, dim=1).indices
                        rank_ok = (torch.arange(26, device=self.device)[None, :] < k[:, None]).float()
                        hits = (mb_opp.gather(1, top) * rank_ok).sum(dim=1)
                        candidates = mb_states[:, KNOWN_CARDS].sum(dim=1) + mb_states[:, UNSEEN_CARDS].sum(dim=1)
                        aux_hits += hits.sum().item()
                        aux_total += k.sum().item()
                        aux_baseline += (k * k / candidates.clamp(min=1)).sum().item()

                if distill is not None:
                    d_idx = torch.randint(len(distill["states"]), (distill_batch,), device=self.device)
                    d_logits, _ = self.model(distill["states"][d_idx], distill["masks"][d_idx])
                    # Illegal actions carry -inf log-probs and zero target mass; clamp keeps 0 * -inf out.
                    d_logp = F.log_softmax(d_logits, dim=-1).clamp(min=-1e4)
                    distill_loss = -(distill["targets"][d_idx] * d_logp).sum(dim=-1).mean()
                    loss = loss + self.cfg.distill_coef * distill_loss
                    total_d_loss += distill_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_a_loss += actor_loss.item()
                total_c_loss += critic_loss.item()
                batches += 1

        aux = {
            "loss": total_aux_loss / batches,
            "precision": aux_hits / max(aux_total, 1.0),
            "baseline": aux_baseline / max(aux_total, 1.0),
        }
        return total_a_loss / batches, total_c_loss / batches, total_d_loss / batches, aux

    def evaluate(self, global_step):
        self.model.eval()
        agent = ModelPolicy(self.model, self.device)
        n = self.cfg.eval_games

        vs_random = play_matches(agent, RandomPolicy(global_step), n, seed=global_step)
        vs_deck = play_matches(agent, DeckOnlyPolicy(global_step), n, seed=global_step + 1)
        vs_greedy = play_matches(agent, GreedyPolicy(global_step), n, seed=global_step + 3)
        greedy_score = vs_greedy["win_rate"] + 0.5 * vs_greedy["draw_rate"]
        blind = ModelPolicy(self.model, self.device, blank_known=True)
        vs_greedy_blind = play_matches(blind, GreedyPolicy(global_step), n, seed=global_step + 3)
        blind_score = vs_greedy_blind["win_rate"] + 0.5 * vs_greedy_blind["draw_rate"]
        self.writer.add_scalar("Eval/WinRate_vs_Random", vs_random["win_rate"], global_step)
        self.writer.add_scalar("Eval/WinRate_vs_DeckOnly", vs_deck["win_rate"], global_step)
        self.writer.add_scalar("Eval/Score_vs_Greedy", greedy_score, global_step)
        self.writer.add_scalar("Eval/Score_vs_Greedy_Blind", blind_score, global_step)
        self.writer.add_scalar("Eval/PenaltyRate", vs_greedy["penalty_rate"], global_step)
        self.writer.add_scalar("Eval/DeepDrawRate", vs_greedy["deep_draw_rate"], global_step)
        self.writer.add_scalar("Eval/PileTakeRate", vs_greedy["pile_take_rate"], global_step)
        self.writer.add_scalar("Eval/MeldPoints", vs_greedy["meld_points"], global_step)
        self.writer.add_scalar("Eval/MeanTurns", vs_greedy["mean_turns"], global_step)
        line = (f"  Eval: vs random {vs_random['win_rate']:.1%} | vs deck-only {vs_deck['win_rate']:.1%} | "
                f"vs greedy {greedy_score:.1%} (blind {blind_score:.1%}) | pile-take {vs_greedy['pile_take_rate']:.1%} | "
                f"meld pts {vs_greedy['meld_points']:.1f} | {vs_greedy['mean_turns']:.1f} turns")

        # Win rate against a snapshot of this policy from frozen_refresh evals ago;
        # above 50% means self-play is still making progress.
        if self.frozen_model is not None:
            vs_frozen = play_matches(agent, ModelPolicy(self.frozen_model, self.device), n, seed=global_step + 2)
            score = vs_frozen["win_rate"] + 0.5 * vs_frozen["draw_rate"]
            self.writer.add_scalar("Eval/Score_vs_Frozen", score, global_step)
            line += f" | vs frozen {score:.1%}"
        if self.evals_run % self.cfg.frozen_refresh == 0:
            self.frozen_model = copy.deepcopy(self.model)
        if self.cfg.search_eval_every and self.evals_run % self.cfg.search_eval_every == 0:
            line += self.evaluate_search(agent, global_step)
        self.evals_run += 1

        self.model.train()
        print(line)

    def evaluate_search(self, plain, global_step):
        cfg = self.cfg
        search = SearchPolicy(self.model, self.device, worlds=cfg.search_worlds,
                              max_actions=cfg.search_actions, seed=global_step)
        t0 = time.time()
        vs_plain = play_matches(search, plain, cfg.search_eval_games, seed=global_step + 4)
        vs_greedy = play_matches(search, GreedyPolicy(global_step), cfg.search_eval_games, seed=global_step + 5)
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
