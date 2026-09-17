import argparse
import os
import sys

import numpy as np
import torch

import rummy_engine
from env.vectorized_env import adapt_obs, blank_known
from models.ppo_network import RummyActorCritic, masked_categorical, sample_categorical


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


# Ace 1, 2-10 face value, court cards 10 (see get_point_value in the engine).
CARD_POINTS = np.array([1 if r == 0 else 10 if r >= 9 else r + 1 for r in range(13)])[np.arange(52) % 13]


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
            actions = sample_categorical(masked_categorical(logits))
        return actions.cpu().numpy()


def play_matches(agent, opponent, num_games, seed=0, num_threads=0, full_game=False, hand_size=7):
    """Agent plays as player 1 in even-indexed games and player 2 in odd ones.

    All games are stepped together in one threaded EnvBatch. By default each
    match is a single round: the winner is the player with the higher score
    when the round ends (someone goes out or the deck runs out). With
    full_game=True matches run until a player reaches the target score.
    """
    if num_threads <= 0:
        num_threads = os.cpu_count() or 1
    batch = rummy_engine.EnvBatch([rummy_engine.RummyEnv(seed + i, 500, 100, hand_size) for i in range(num_games)],
                                  num_threads)
    agent_player = np.where(np.arange(num_games) % 2 == 0, 1, 2).astype(np.int32)
    alive = np.ones(num_games, dtype=bool)

    wins = draws = 0
    agent_penalties = 0
    agent_draws = agent_deep_draws = agent_pile_available = 0
    steps = np.zeros(num_games, dtype=np.int64)
    # Agent's score sampled before its final step: points laid on the table,
    # before the leftover-hand subtraction at the end of the round.
    meld_points = np.zeros(num_games, dtype=np.float64)
    final_scores = np.zeros((num_games, 2), dtype=np.float64)

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
        if not full_game:
            meld_points[live] = scores[live, agent_player[live] - 1]
        rewards, dones, round_ends = batch.step(actions)
        steps[live] += 1

        over = dones if full_game else (dones | round_ends)
        finished = live[over[live]]
        if len(finished):
            scores = batch.scores()
            final_scores[finished] = scores[finished]
            own = scores[finished, agent_player[finished] - 1]
            opp = scores[finished, 2 - agent_player[finished]]
            penalised = batch.penalised()[finished]
            # Breaking the pile-draw obligation forfeits the game.
            lost_by_penalty = penalised == agent_player[finished]
            won_by_penalty = (penalised != 0) & ~lost_by_penalty
            agent_penalties += int(lost_by_penalty.sum())
            wins += int((won_by_penalty | ((penalised == 0) & (own > opp))).sum())
            draws += int(((penalised == 0) & (own == opp)).sum())
            alive[finished] = False
            batch.halt(~alive)

    return {
        "win_rate": wins / num_games,
        "draw_rate": draws / num_games,
        "penalty_rate": agent_penalties / num_games,
        "deep_draw_rate": agent_deep_draws / max(agent_draws, 1),
        "pile_take_rate": agent_deep_draws / max(agent_pile_available, 1),
        "meld_points": float(meld_points.mean()) if not full_game
                       else float(final_scores[np.arange(num_games), agent_player - 1].mean()),
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
    parser.add_argument("--full", action="store_true",
                        help="play full games to the target score instead of single rounds (about 10x slower)")
    args = parser.parse_args()
    full = args.full

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = ModelPolicy(load_model(args.checkpoints[0], device), device, args.greedy)
    random_policy = RandomPolicy(args.seed)
    deck_only = DeckOnlyPolicy(args.seed)
    greedy = GreedyPolicy(args.seed)

    def score(s):
        return s["win_rate"] + 0.5 * s["draw_rate"]

    unit = "full games to the target score" if full else "single rounds"
    points_col = "score" if full else "meld pts"
    print(f"baseline: {os.path.basename(args.checkpoints[0])}   matches are {unit}\n"
          f"(scores count a draw as half a win; pile-take, {points_col} and turns are from games vs greedy)")
    print(f"{'checkpoint':<28} {'vs random':>9} {'vs deck':>8} {'vs greedy':>9} {'(blind)':>8} {'vs base':>8} "
          f"{'penalty':>8} {'pile-take':>10} {points_col:>9} {'turns':>6}")
    for path in args.checkpoints:
        model = load_model(path, device)
        agent = ModelPolicy(model, device, args.greedy)
        blind = ModelPolicy(model, device, args.greedy, blank_known=True)
        vs_random = play_matches(agent, random_policy, args.games, args.seed, full_game=full)
        vs_deck = play_matches(agent, deck_only, args.games, args.seed + 1, full_game=full)
        vs_greedy = play_matches(agent, greedy, args.games, args.seed + 2, full_game=full)
        vs_greedy_blind = play_matches(blind, greedy, args.games, args.seed + 2, full_game=full)
        vs_base = play_matches(agent, baseline, args.games, args.seed + 3, full_game=full)
        print(f"{os.path.basename(path):<28} {score(vs_random):>9.1%} {score(vs_deck):>8.1%} "
              f"{score(vs_greedy):>9.1%} {score(vs_greedy_blind):>8.1%} {score(vs_base):>8.1%} "
              f"{vs_greedy['penalty_rate']:>8.1%} {vs_greedy['pile_take_rate']:>10.1%} "
              f"{vs_greedy['meld_points']:>9.1f} {vs_greedy['mean_turns']:>6.1f}")


if __name__ == "__main__":
    sys.exit(main())
