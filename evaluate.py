import argparse
import os
import sys

import numpy as np
import torch

import rummy_engine
from env.vectorized_env import adapt_obs, blank_known
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
            obs = blank_known(obs.copy())
        obs = adapt_obs(obs, self.model.obs_dim)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        logits, _ = self.model(obs_t, mask_t)
        if self.greedy:
            actions = logits.argmax(dim=-1)
        else:
            actions = torch.distributions.Categorical(logits=logits).sample()
        return actions.cpu().numpy()


def play_matches(agent, opponent, num_games, seed=0, num_threads=0):
    """Agent plays as player 1 in even-indexed games and player 2 in odd ones.

    All games are stepped together in one threaded EnvBatch. Terminal rewards
    are from the acting player's perspective: >0 the actor won, <0 the actor
    lost, 0 a draw.
    """
    if num_threads <= 0:
        num_threads = os.cpu_count() or 1
    batch = rummy_engine.EnvBatch([rummy_engine.RummyEnv(seed + i) for i in range(num_games)], num_threads)
    agent_player = np.where(np.arange(num_games) % 2 == 0, 1, 2).astype(np.int32)
    alive = np.ones(num_games, dtype=bool)

    wins = draws = 0
    agent_penalties = 0
    agent_draws = agent_deep_draws = agent_pile_available = 0
    steps = np.zeros(num_games, dtype=np.int64)
    # Agent's score sampled before its final step: meld points, excluding the
    # opponent's leftover hand value that settle_terminal() adds at game end.
    meld_points = np.zeros(num_games, dtype=np.float64)

    while alive.any():
        states, masks, players = batch.observe()
        live = np.flatnonzero(alive)
        agent_turn = players[live] == agent_player[live]
        agent_idx = live[agent_turn]
        opp_idx = live[~agent_turn]

        actions = np.zeros(num_games, dtype=np.int64)
        if len(agent_idx):
            if getattr(agent, "needs_env", False):
                actions[agent_idx] = agent.act_envs([batch.get(int(i)) for i in agent_idx])
            else:
                actions[agent_idx] = agent.act(states[agent_idx], masks[agent_idx])
        if len(opp_idx):
            if getattr(opponent, "needs_env", False):
                actions[opp_idx] = opponent.act_envs([batch.get(int(i)) for i in opp_idx])
            else:
                actions[opp_idx] = opponent.act(states[opp_idx], masks[opp_idx])

        draw_phase = states[agent_idx, -3] == 0.0
        agent_draws += int(draw_phase.sum())
        agent_deep_draws += int((draw_phase & (actions[agent_idx] > 0)).sum())
        agent_pile_available += int((draw_phase & masks[agent_idx, 1:].any(axis=1)).sum())

        scores = batch.scores()
        meld_points[live] = scores[live, agent_player[live] - 1]
        rewards, dones = batch.step(actions)
        steps[live] += 1

        finished = live[dones[live]]
        if len(finished):
            actor_is_agent = players[finished] == agent_player[finished]
            r = rewards[finished]
            draws += int((r == 0).sum())
            wins += int(((r != 0) & ((r > 0) == actor_is_agent)).sum())
            # A -50 with cards still in the deck is the meld-failure penalty; at deck
            # exhaustion -50 can also be a legitimate score difference.
            end_states = batch.observe()[0]
            agent_penalties += int((actor_is_agent & (r == -50) & (end_states[finished, -2] < 1.0)).sum())
            alive[finished] = False

    return {
        "win_rate": wins / num_games,
        "draw_rate": draws / num_games,
        "penalty_rate": agent_penalties / num_games,
        "deep_draw_rate": agent_deep_draws / max(agent_draws, 1),
        "pile_take_rate": agent_deep_draws / max(agent_pile_available, 1),
        "meld_points": float(meld_points.mean()),
        "mean_turns": float(steps.mean()) / 2,
    }


def load_model(path, device):
    model = RummyActorCritic.from_state_dict(torch.load(path, map_location=device)).to(device)
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
