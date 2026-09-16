import argparse
import os
import sys

import numpy as np
import torch

import rummy_engine
from config import PPOConfig
from env.vectorized_env import KNOWN_CARDS
from models.ppo_network import RummyActorCritic


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, mask):
        probs = mask.astype(np.float64)
        probs /= probs.sum(axis=1, keepdims=True)
        return np.array([self.rng.choice(mask.shape[1], p=p) for p in probs])


class DeckOnlyPolicy:
    """Always draws from the deck and discards at random: never penalised, so
    games run to the deck or to someone going out."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, mask):
        actions = np.empty(len(obs), dtype=np.int64)
        for i in range(len(obs)):
            legal = np.flatnonzero(mask[i])
            actions[i] = 0 if obs[i, -3] == 0.0 and mask[i, 0] else self.rng.choice(legal)
        return actions


CARD_POINTS = np.array([15 if r == 0 else 10 if r >= 9 else 5 for r in range(52)])[np.arange(52) % 13]


class GreedyPolicy:
    """Deepest legal pile draw (else deck); discard the highest-value legal card."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, mask):
        actions = np.empty(len(obs), dtype=np.int64)
        for i in range(len(obs)):
            legal = np.flatnonzero(mask[i])
            if obs[i, -3] == 0.0:
                pile = legal[legal > 0]
                actions[i] = pile.min() if len(pile) else 0
            else:
                points = CARD_POINTS[legal - 53]
                best = legal[points == points.max()]
                actions[i] = self.rng.choice(best)
        return actions


class ModelPolicy:
    def __init__(self, model, device, greedy=False, blank_known=False):
        self.model = model
        self.device = device
        self.greedy = greedy
        self.blank_known = blank_known

    @torch.no_grad()
    def act(self, obs, mask):
        if self.blank_known:
            obs = obs.copy()
            obs[:, KNOWN_CARDS] = 0.0
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
    agent_draws = agent_deep_draws = agent_pile_available = 0
    game_lengths = np.zeros(num_games, dtype=np.int64)
    # Agent's score sampled before its final step: meld points, excluding the
    # opponent's leftover hand value that settle_terminal() adds at game end.
    meld_points = np.zeros(num_games, dtype=np.float64)

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

        draw_phase = obs[:, -3] == 0.0
        agent_draws += int((agent_turn & draw_phase).sum())
        agent_deep_draws += int((agent_turn & draw_phase & (actions > 0)).sum())
        agent_pile_available += int((agent_turn & draw_phase & mask[:, 1:].any(axis=1)).sum())

        for j, i in enumerate(live):
            meld_points[i] = envs[i].get_score(int(agent_player[i]))
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
            # A -50 with cards still in the deck is the meld-failure penalty; at deck
            # exhaustion -50 can also be a legitimate score difference.
            if actor_is_agent and reward == -50 and envs[i].get_state()[-2] < 1.0:
                agent_penalties += 1

    return {
        "win_rate": wins / num_games,
        "draw_rate": draws / num_games,
        "penalty_rate": agent_penalties / num_games,
        "deep_draw_rate": agent_deep_draws / max(agent_draws, 1),
        "pile_take_rate": agent_deep_draws / max(agent_pile_available, 1),
        "meld_points": float(meld_points.mean()),
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
    deck_only = DeckOnlyPolicy(args.seed)
    greedy = GreedyPolicy(args.seed)

    def score(s):
        return s["win_rate"] + 0.5 * s["draw_rate"]

    print(f"baseline: {os.path.basename(args.checkpoints[0])}   "
          f"(scores count a draw as half a win; pile-take, meld pts and turns are from games vs greedy)")
    print(f"{'checkpoint':<28} {'vs random':>9} {'vs deck':>8} {'vs greedy':>9} {'(blind)':>8} {'vs base':>8} "
          f"{'penalty':>8} {'pile-take':>10} {'meld pts':>9} {'turns':>6}")
    for path in args.checkpoints:
        model = load_model(path, device)
        agent = ModelPolicy(model, device, args.greedy)
        blind = ModelPolicy(model, device, args.greedy, blank_known=True)
        vs_random = play_matches(agent, random_policy, args.games, args.seed)
        vs_deck = play_matches(agent, deck_only, args.games, args.seed + 1)
        vs_greedy = play_matches(agent, greedy, args.games, args.seed + 2)
        vs_greedy_blind = play_matches(blind, greedy, args.games, args.seed + 2)
        vs_base = play_matches(agent, baseline, args.games, args.seed + 3)
        print(f"{os.path.basename(path):<28} {score(vs_random):>9.1%} {score(vs_deck):>8.1%} "
              f"{score(vs_greedy):>9.1%} {score(vs_greedy_blind):>8.1%} {score(vs_base):>8.1%} "
              f"{vs_greedy['penalty_rate']:>8.1%} {vs_greedy['pile_take_rate']:>10.1%} "
              f"{vs_greedy['meld_points']:>9.1f} {vs_greedy['mean_turns']:>6.1f}")


if __name__ == "__main__":
    sys.exit(main())
