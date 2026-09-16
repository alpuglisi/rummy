import os

import numpy as np
import torch

import rummy_engine

# Observation slice holding the opponent's publicly known cards (see rummy_env.h).
KNOWN_CARDS = slice(4 * 52, 5 * 52)


class VectorizedRummyEnv:
    """Steps all games inside the C++ engine in one call; no worker processes.

    With blank_known_prob > 0, a fraction of games hide the opponent-known-cards
    channel for their whole duration, so the policy learns to play with or
    without that information.
    """

    def __init__(self, num_envs, num_threads=0, seed=None, blank_known_prob=0.0):
        if seed is None:
            seed = int(np.random.randint(0, 2**31))
        if num_threads <= 0:
            num_threads = min(num_envs, os.cpu_count() or 1)
        self.num_envs = num_envs
        self.blank_known_prob = blank_known_prob
        self.rng = np.random.default_rng(seed)
        self.blank = np.zeros(num_envs, dtype=bool)
        self._env = rummy_engine.VectorizedRummyEnv(num_envs, seed, num_threads)

    def _resample_blank(self, which):
        if self.blank_known_prob > 0:
            self.blank[which] = self.rng.random(int(np.count_nonzero(which))) < self.blank_known_prob

    def _apply_blank(self, states):
        if self.blank.any():
            states[self.blank, KNOWN_CARDS] = 0.0
        return states

    def reset(self):
        states, masks = self._env.reset()
        self._resample_blank(np.ones(self.num_envs, dtype=bool))
        return torch.from_numpy(self._apply_blank(states)), torch.from_numpy(masks)

    def step(self, actions):
        states, masks, rewards, dones = self._env.step(actions.detach().cpu().numpy())
        if dones.any():
            self._resample_blank(dones)
        return (torch.from_numpy(self._apply_blank(states)), torch.from_numpy(masks),
                torch.from_numpy(rewards), torch.from_numpy(dones))

    def close(self):
        pass
