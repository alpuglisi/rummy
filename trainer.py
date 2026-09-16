import copy
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from env.vectorized_env import VectorizedRummyEnv
from models.ppo_network import RummyActorCritic
from config import PPOConfig
from evaluate import DeckOnlyPolicy, ModelPolicy, RandomPolicy, play_matches

class RolloutBuffer:
    def __init__(self, cfg: PPOConfig, device: torch.device):
        self.states = torch.zeros((cfg.num_steps, cfg.num_envs, cfg.obs_dim), dtype=torch.float32).to(device)
        self.masks = torch.zeros((cfg.num_steps, cfg.num_envs, cfg.action_dim), dtype=torch.bool).to(device)
        self.actions = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.long).to(device)
        self.logprobs = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.float32).to(device)
        self.values = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.float32).to(device)
        self.dones = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.bool).to(device)
        self.step = 0
        self.device = device

    def store(self, state, mask, action, logprob, reward, value, done):
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
        
        self.envs = VectorizedRummyEnv(config.num_envs, config.env_threads)
        self.model = RummyActorCritic(obs_dim=config.obs_dim, action_dim=config.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.buffer = RolloutBuffer(config, self.device)
        
        self.writer = SummaryWriter(log_dir="runs/rummy_ppo")
        self.frozen_model = None
        self.evals_run = 0
        
    def train(self):
        state, mask = self.envs.reset()
        state = state.to(self.device)
        mask = mask.to(self.device)
        done = torch.zeros(self.cfg.num_envs, dtype=torch.bool).to(self.device)
        
        global_step = 0
        update = 0
        while global_step < self.cfg.total_timesteps:
            for _ in range(self.cfg.num_steps):
                with torch.no_grad():
                    action, logprob, value = self.model.get_action(state, mask)
                
                next_state, next_mask, reward, next_done = self.envs.step(action)
                reward = reward * self.cfg.reward_scale

                self.buffer.store(state, mask, action, logprob, reward, value, done)
                
                state = next_state.to(self.device)
                mask = next_mask.to(self.device)
                done = next_done.to(self.device)
                global_step += self.cfg.num_envs

            with torch.no_grad():
                _, next_value = self.model(state, mask)
                next_value = next_value.squeeze(-1)
            
            advantages, returns = self.buffer.compute_advantages(
                next_value, state, done, self.cfg.gamma, self.cfg.gae_lambda
            )
            
            actor_loss, critic_loss = self.optimize(advantages, returns)
            
            self.writer.add_scalar("Loss/Actor", actor_loss, global_step)
            self.writer.add_scalar("Loss/Critic", critic_loss, global_step)
            self.writer.add_scalar("Reward/Average_Return", returns.mean().item(), global_step)
            
            self.buffer.clear()
            print(f"Global Step: {global_step} / {self.cfg.total_timesteps}")

            update += 1
            if update % self.cfg.eval_interval == 0:
                self.evaluate(global_step)
            
            current_millions = global_step // 1_000_000

            
            previous_millions = (global_step - (self.cfg.num_envs * self.cfg.num_steps)) // 1_000_000

            
            if current_millions > previous_millions:

            
                self.save_checkpoint(f"checkpoints/ppo_rummy_{global_step}.pth")

            
                print(f"Checkpoint saved at step {global_step}!")

    def optimize(self, advantages, returns):
        b_states = self.buffer.states.view(-1, self.cfg.obs_dim)
        b_masks = self.buffer.masks.view(-1, self.cfg.action_dim)
        b_actions = self.buffer.actions.view(-1)
        b_logprobs = self.buffer.logprobs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        num_samples = b_states.shape[0]
        total_a_loss = 0
        total_c_loss = 0
        batches = 0

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

                logits, new_values = self.model(mb_states, mb_masks)
                dist = torch.distributions.Categorical(logits=logits)

                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                logratio = new_logprobs - mb_old_logprobs
                ratio = logratio.exp()

                pg_loss1 = mb_advantages * ratio
                pg_loss2 = mb_advantages * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
                actor_loss = -torch.min(pg_loss1, pg_loss2).mean()

                critic_loss = F.mse_loss(new_values.squeeze(-1), mb_returns)

                loss = actor_loss + (self.cfg.vf_coef * critic_loss) - (self.cfg.ent_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_a_loss += actor_loss.item()
                total_c_loss += critic_loss.item()
                batches += 1
                
        return total_a_loss / batches, total_c_loss / batches

    def evaluate(self, global_step):
        self.model.eval()
        agent = ModelPolicy(self.model, self.device)
        n = self.cfg.eval_games

        vs_random = play_matches(agent, RandomPolicy(global_step), n, seed=global_step)
        vs_deck = play_matches(agent, DeckOnlyPolicy(global_step), n, seed=global_step + 1)
        self.writer.add_scalar("Eval/WinRate_vs_Random", vs_random["win_rate"], global_step)
        self.writer.add_scalar("Eval/WinRate_vs_DeckOnly", vs_deck["win_rate"], global_step)
        self.writer.add_scalar("Eval/DrawRate_vs_DeckOnly", vs_deck["draw_rate"], global_step)
        self.writer.add_scalar("Eval/PenaltyRate", vs_deck["penalty_rate"], global_step)
        self.writer.add_scalar("Eval/DeepDrawRate", vs_deck["deep_draw_rate"], global_step)
        self.writer.add_scalar("Eval/MeldPoints", vs_deck["meld_points"], global_step)
        self.writer.add_scalar("Eval/MeanTurns", vs_deck["mean_turns"], global_step)
        line = (f"  Eval: vs random {vs_random['win_rate']:.1%} | vs deck-only {vs_deck['win_rate']:.1%} "
                f"(draws {vs_deck['draw_rate']:.1%}) | deep-draw {vs_deck['deep_draw_rate']:.1%} | "
                f"meld pts {vs_deck['meld_points']:.1f} | {vs_deck['mean_turns']:.1f} turns")

        # Win rate against a snapshot of this policy from frozen_refresh evals ago;
        # above 50% means self-play is still making progress.
        if self.frozen_model is not None:
            vs_frozen = play_matches(agent, ModelPolicy(self.frozen_model, self.device), n, seed=global_step + 2)
            score = vs_frozen["win_rate"] + 0.5 * vs_frozen["draw_rate"]
            self.writer.add_scalar("Eval/Score_vs_Frozen", score, global_step)
            line += f" | vs frozen {score:.1%}"
        if self.evals_run % self.cfg.frozen_refresh == 0:
            self.frozen_model = copy.deepcopy(self.model)
        self.evals_run += 1

        self.model.train()
        print(line)

    def save_checkpoint(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.model.state_dict(), path)

if __name__ == "__main__":
    cfg = PPOConfig()
    trainer = PPOTrainer(cfg)
    trainer.train()
