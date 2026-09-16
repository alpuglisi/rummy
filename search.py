import numpy as np
import torch


class SearchPolicy:
    """Determinized Monte Carlo search on top of a trained policy.

    For each decision: sample `worlds` redeals of the hidden cards, play every
    candidate action out to the end of the game in each world with the model
    driving both players, and pick the action with the best mean outcome from
    the searching player's perspective. Candidate actions are the model's
    `max_actions` most likely legal moves.
    """

    needs_env = True

    def __init__(self, model, device, worlds=16, max_actions=4, seed=0):
        self.model = model
        self.device = device
        self.worlds = worlds
        self.max_actions = max_actions
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def _policy(self, obs, mask):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=self.device)
        logits, _ = self.model(obs_t, mask_t)
        return logits

    def _sample(self, obs, mask):
        return torch.distributions.Categorical(logits=self._policy(obs, mask)).sample().cpu().numpy()

    def evaluate_actions(self, envs):
        """Return, per env, a dict {action: (mean_outcome, n_rollouts)} for its candidates."""
        obs = np.stack([e.get_state() for e in envs])
        mask = np.stack([e.get_legal_actions() for e in envs]).astype(bool)
        probs = torch.softmax(self._policy(obs, mask), dim=-1).cpu().numpy()

        candidates = []
        for i in range(len(envs)):
            legal = np.flatnonzero(mask[i])
            order = legal[np.argsort(-probs[i, legal])]
            candidates.append(order[: self.max_actions])

        # One rollout per (env, world, candidate); all advanced in lockstep.
        clones, tags, agents = [], [], []
        totals = [{int(a): [0.0, 0] for a in c} for c in candidates]
        for i, env in enumerate(envs):
            agent = env.get_current_player()
            for w in range(self.worlds):
                world = env.clone()
                world.randomize_hidden(int(self.rng.integers(0, 2**31)))
                for a in candidates[i]:
                    sim = world.clone()
                    reward, done = sim.step(int(a))
                    if done:
                        totals[i][int(a)][0] += reward   # the searcher acted, so reward is already ours
                        totals[i][int(a)][1] += 1
                    else:
                        clones.append(sim)
                        tags.append((i, int(a)))
                        agents.append(agent)

        while clones:
            obs = np.stack([c.get_state() for c in clones])
            mask = np.stack([c.get_legal_actions() for c in clones]).astype(bool)
            actions = self._sample(obs, mask)
            survivors, survivor_tags, survivor_agents = [], [], []
            for c, (i, a), agent, act in zip(clones, tags, agents, actions):
                actor = c.get_current_player()
                reward, done = c.step(int(act))
                if done:
                    outcome = reward if actor == agent else -reward
                    totals[i][a][0] += outcome
                    totals[i][a][1] += 1
                else:
                    survivors.append(c)
                    survivor_tags.append((i, a))
                    survivor_agents.append(agent)
            clones, tags, agents = survivors, survivor_tags, survivor_agents

        return [{a: (s / n, n) for a, (s, n) in t.items()} for t in totals]

    def act_envs(self, envs):
        stats = self.evaluate_actions(envs)
        return np.array([max(s, key=lambda a: s[a][0]) for s in stats], dtype=np.int64)
