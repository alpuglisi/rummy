import os

import numpy as np
import torch

import rummy_engine

# Observation layout (see rummy_env.h): six 52-card channels then six scalars.
KNOWN_CARDS = slice(4 * 52, 5 * 52)    # opponent's publicly known hand cards
UNSEEN_CARDS = slice(5 * 52, 6 * 52)   # deck or opponent's unknown hand
CHANNELS_END = 6 * 52


def blank_known(states):
    """Hide the opponent-known channel in place, folding those cards into
    'unseen' so the view matches a player who did not track the opponent's picks."""
    states[:, UNSEEN_CARDS] = np.maximum(states[:, UNSEEN_CARDS], states[:, KNOWN_CARDS])
    states[:, KNOWN_CARDS] = 0.0
    return states


def adapt_obs(states, obs_dim):
    """Project the engine observation onto an older layout (a checkpoint
    trained without the unseen channel expects the five channels + scalars).
    Works on numpy arrays and torch tensors."""
    if states.shape[-1] == obs_dim:
        return states
    if obs_dim == 5 * 52 + 6 and states.shape[-1] == CHANNELS_END + 6:
        if isinstance(states, np.ndarray):
            return np.concatenate([states[..., :5 * 52], states[..., CHANNELS_END:]], axis=-1)
        return torch.cat([states[..., :5 * 52], states[..., CHANNELS_END:]], dim=-1)
    raise ValueError(f"cannot adapt observation of width {states.shape[-1]} to a model expecting {obs_dim}")


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
            states[self.blank] = blank_known(states[self.blank])
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

    def clone(self, i):
        """Independent copy of live game i, for search."""
        return self._env.get(int(i))

    def current_players(self):
        return torch.from_numpy(self._env.current_players())

    def opponent_hands(self):
        """Ground-truth cards of the player not to move, [N,52] bool. Training target only."""
        return torch.from_numpy(self._env.opponent_hands())

    def close(self):
        pass
