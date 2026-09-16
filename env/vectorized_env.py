import os

import numpy as np
import torch

import rummy_engine


class VectorizedRummyEnv:
    """Steps all games inside the C++ engine in one call; no worker processes."""

    def __init__(self, num_envs, num_threads=1, seed=None):
        if seed is None:
            seed = int(np.random.randint(0, 2**31))
        if num_threads <= 0:
            num_threads = min(num_envs, os.cpu_count() or 1)
        self.num_envs = num_envs
        self._env = rummy_engine.VectorizedRummyEnv(num_envs, seed, num_threads)

    def reset(self):
        states, masks = self._env.reset()
        return torch.from_numpy(states), torch.from_numpy(masks)

    def step(self, actions):
        states, masks, rewards, dones = self._env.step(actions.detach().cpu().numpy())
        return (torch.from_numpy(states), torch.from_numpy(masks),
                torch.from_numpy(rewards), torch.from_numpy(dones))

    def close(self):
        pass
