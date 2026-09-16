import argparse
import time

import torch

from evaluate import GreedyPolicy, ModelPolicy, load_model, play_matches
from search import SearchPolicy


def score(s):
    return s["win_rate"] + 0.5 * s["draw_rate"]


def main():
    parser = argparse.ArgumentParser(description="Head-to-head: Monte Carlo search on a checkpoint vs the plain policy.")
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--actions", type=int, default=4, help="candidate actions per decision")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    plain = ModelPolicy(model, device)
    search = SearchPolicy(model, device, worlds=args.worlds, max_actions=args.actions, seed=args.seed)

    t0 = time.time()
    vs_plain = play_matches(search, plain, args.games, args.seed)
    elapsed = time.time() - t0
    print(f"search({args.worlds} worlds x {args.actions} actions) vs plain policy: {score(vs_plain):.1%} "
          f"over {args.games} games  [{elapsed:.0f}s, ~{elapsed / args.games:.1f}s per game]")

    vs_greedy = play_matches(search, GreedyPolicy(args.seed), args.games, args.seed + 1)
    plain_vs_greedy = play_matches(plain, GreedyPolicy(args.seed), args.games * 5, args.seed + 1)
    print(f"search vs greedy: {score(vs_greedy):.1%}   plain vs greedy: {score(plain_vs_greedy):.1%}")


if __name__ == "__main__":
    main()
