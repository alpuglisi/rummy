import argparse
import os
import sys

import numpy as np
import torch

import rummy_engine
from config import PPOConfig
from models.ppo_network import RummyActorCritic


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, mask):
        probs = mask.astype(np.float64)
        probs /= probs.sum(axis=1, keepdims=True)
        return np.array([self.rng.choice(mask.shape[1], p=p) for p in probs])


class ModelPolicy:
    def __init__(self, model, device, greedy=False):
        self.model = model
        self.device = device
        self.greedy = greedy

    @torch.no_grad()
    def act(self, obs, mask):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        logits, _ = self.model(obs_t, mask_t)
        if self.greedy:
            actions = logits.argmax(dim=-1)
        else:
            actions = torch.distributions.Categorical(logits=logits).sample()
        return actions.cpu().numpy()


def play_matches(agent, opponent, num_games, seed=0):
    """Agent plays as player 1 in even-indexed games and player 2 in odd ones.

    A game is a strict draw/discard alternation, so the player to act at step k
    is 1 when (k // 2) is even and 2 otherwise. Terminal rewards are from the
    acting player's perspective: >0 the actor won, <0 the actor lost, 0 a draw.
    """
    envs = [rummy_engine.RummyEnv(seed + i) for i in range(num_games)]
    for env in envs:
        env.reset()
    agent_player = np.where(np.arange(num_games) % 2 == 0, 1, 2)
    step_idx = np.zeros(num_games, dtype=np.int64)
    alive = np.ones(num_games, dtype=bool)

    wins = draws = 0
    agent_penalties = 0
    game_lengths = np.zeros(num_games, dtype=np.int64)

    while alive.any():
        live = np.flatnonzero(alive)
        obs = np.stack([envs[i].get_state() for i in live])
        mask = np.stack([envs[i].get_legal_actions() for i in live])
        acting = np.where((step_idx[live] // 2) % 2 == 0, 1, 2)
        agent_turn = acting == agent_player[live]

        actions = np.empty(len(live), dtype=np.int64)
        if agent_turn.any():
            actions[agent_turn] = agent.act(obs[agent_turn], mask[agent_turn])
        if (~agent_turn).any():
            actions[~agent_turn] = opponent.act(obs[~agent_turn], mask[~agent_turn])

        for j, i in enumerate(live):
            reward, done = envs[i].step(int(actions[j]))
            step_idx[i] += 1
            if not done:
                continue
            alive[i] = False
            game_lengths[i] = step_idx[i]
            actor_is_agent = agent_turn[j]
            if reward == 0:
                draws += 1
            elif (reward > 0) == actor_is_agent:
                wins += 1
            if actor_is_agent and reward == -50:
                agent_penalties += 1

    return {
        "win_rate": wins / num_games,
        "draw_rate": draws / num_games,
        "penalty_rate": agent_penalties / num_games,
        "mean_turns": float(game_lengths.mean()) / 2,
    }


def load_model(path, device):
    cfg = PPOConfig()
    model = RummyActorCritic(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Score checkpoints against a random opponent and a baseline checkpoint.")
    parser.add_argument("checkpoints", nargs="+", help="checkpoint .pth files (first one is the baseline opponent)")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true", help="argmax actions instead of sampling")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = ModelPolicy(load_model(args.checkpoints[0], device), device, args.greedy)
    random_policy = RandomPolicy(args.seed)

    print(f"{'checkpoint':<40} {'vs random':>10} {'vs baseline':>12} {'draws':>7} {'penalty':>8} {'turns':>6}")
    for path in args.checkpoints:
        agent = ModelPolicy(load_model(path, device), device, args.greedy)
        vs_random = play_matches(agent, random_policy, args.games, args.seed)
        vs_base = play_matches(agent, baseline, args.games, args.seed + 1)
        print(f"{os.path.basename(path):<40} {vs_random['win_rate']:>10.1%} {vs_base['win_rate']:>12.1%} "
              f"{vs_base['draw_rate']:>7.1%} {vs_random['penalty_rate']:>8.1%} {vs_random['mean_turns']:>6.1f}")


if __name__ == "__main__":
    sys.exit(main())
