import os

import numpy as np
import torch

import rummy_engine
from env.vectorized_env import adapt_obs


class SearchPolicy:
    """Determinized Monte Carlo search on top of a trained policy.

    For each decision: sample `worlds` redeals of the hidden cards, play every
    candidate action out to the end of the game in each world with the model
    driving both players, and pick the action with the best mean outcome from
    the searching player's perspective. Candidate actions are the model's
    `max_actions` most likely legal moves. All rollouts for all decisions are
    stepped together in one threaded EnvBatch.
    """

    needs_env = True

    def __init__(self, model, device, worlds=16, max_actions=4, seed=0, num_threads=0, belief=False):
        self.model = model
        self.device = device
        self.worlds = worlds
        self.max_actions = max_actions
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
        return torch.sigmoid(aux).cpu().numpy().astype(np.float32)

    def reset_stats(self):
        self.decisions = 0
        self.agreements = 0          # search picked the model's own top choice
        self.value_gap = 0.0         # mean outcome of search's pick minus the model's pick
        self.rollouts = 0
        self.rollout_steps = 0       # engine steps played across all rollouts (they run to game end)

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
        sims, sim_env, sim_action, sim_agent = [], [], [], []
        for i, env in enumerate(envs):
            agent = env.get_current_player()
            for _ in range(self.worlds):
                world = env.clone()
                seed = int(self.rng.integers(0, 2**31))
                if beliefs is not None:
                    world.randomize_hidden_weighted(seed, beliefs[i])
                else:
                    world.randomize_hidden(seed)
                for a in candidates[i]:
                    sims.append(world.clone())
                    sim_env.append(i)
                    sim_action.append(int(a))
                    sim_agent.append(agent)
        sim_env = np.array(sim_env)
        sim_action = np.array(sim_action, dtype=np.int64)
        sim_agent = np.array(sim_agent, dtype=np.int32)

        batch = rummy_engine.EnvBatch(sims, self.num_threads)
        n = batch.size
        outcome = np.zeros(n, dtype=np.float64)
        actor = np.full(n, sim_agent, dtype=np.int32)
        rewards, dones = batch.step(sim_action)
        finished = dones.copy()
        outcome[finished] = rewards[finished]   # the searcher acted, so the reward is already ours

        self.rollouts += n
        self.rollout_steps += n
        alive = ~finished
        while alive.any():
            states, masks, players = batch.observe()
            idx = np.flatnonzero(alive)
            self.rollout_steps += len(idx)
            actions = np.zeros(n, dtype=np.int64)
            actions[idx] = self._sample(states[idx], masks[idx])
            actor[idx] = players[idx]
            rewards, dones = batch.step(actions)
            just_done = idx[dones[idx]]
            sign = np.where(actor[just_done] == sim_agent[just_done], 1.0, -1.0)
            outcome[just_done] = rewards[just_done] * sign
            alive[just_done] = False

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
