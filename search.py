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

    def __init__(self, model, device, worlds=16, max_actions=4, seed=0, num_threads=0):
        self.model = model
        self.device = device
        self.worlds = worlds
        self.max_actions = max_actions
        self.rng = np.random.default_rng(seed)
        self.num_threads = num_threads if num_threads > 0 else (os.cpu_count() or 1)

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

        # One rollout per (env, world, candidate). Worlds are built once per env
        # and copied per candidate so every candidate faces the same redeals.
        sims, sim_env, sim_action, sim_agent = [], [], [], []
        for i, env in enumerate(envs):
            agent = env.get_current_player()
            for _ in range(self.worlds):
                world = env.clone()
                world.randomize_hidden(int(self.rng.integers(0, 2**31)))
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

        alive = ~finished
        while alive.any():
            states, masks, players = batch.observe()
            idx = np.flatnonzero(alive)
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
        return np.array([max(s, key=lambda a: s[a][0]) for s in stats], dtype=np.int64)
