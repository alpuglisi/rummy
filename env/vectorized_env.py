import os

import numpy as np
import torch

import rummy_engine

# Observation layout (see rummy_env.h): six 52-card channels, the turn
# history block, then eight scalars (own score / target, opponent score /
# target, own hand size, opponent hand size, score difference, turn phase,
# deck fraction, required-meld flag).
KNOWN_CARDS = slice(4 * 52, 5 * 52)    # opponent's publicly known hand cards
UNSEEN_CARDS = slice(5 * 52, 6 * 52)   # deck or opponent's unknown hand
CHANNELS_END = 6 * 52
HISTORY_LEN = rummy_engine.HISTORY_LEN
EVENT_DIM = rummy_engine.EVENT_DIM
HISTORY = slice(CHANNELS_END, CHANNELS_END + HISTORY_LEN * EVENT_DIM)
SCORE_SCALARS = 2
SCALARS = 8
OBS_DIM = rummy_engine.OBS_SPACE_SIZE
AUX_DIM = rummy_engine.AUX_SPACE_SIZE
assert OBS_DIM == CHANNELS_END + HISTORY_LEN * EVENT_DIM + SCALARS

# Event encoding inside the history block: [by opponent, deck draw, pile take,
# discard, rank one-hot (13), suit one-hot (4), cards taken / 10].
EVENT_OPPONENT, EVENT_PILE_TAKE, EVENT_CARD = 0, 2, slice(4, 4 + 13 + 4)


def blank_known(states):
    """Hide the opponent-known channel in place, folding those cards into
    'unseen' so the view matches a player who did not track the opponent's
    picks. The card of the opponent's pile-take events in the history is
    hidden for the same reason (the take itself, and its size, stay visible)."""
    states[:, UNSEEN_CARDS] = np.maximum(states[:, UNSEEN_CARDS], states[:, KNOWN_CARDS])
    states[:, KNOWN_CARDS] = 0.0
    hist = states[:, HISTORY].reshape(len(states), HISTORY_LEN, EVENT_DIM)
    opp_take = (hist[:, :, EVENT_OPPONENT] > 0.5) & (hist[:, :, EVENT_PILE_TAKE] > 0.5)
    hist[opp_take, EVENT_CARD] = 0.0
    states[:, HISTORY] = hist.reshape(len(states), -1)
    return states


def adapt_obs(states, obs_dim):
    """Project the engine observation onto an older layout: checkpoints from
    before the turn history expect six channels + eight scalars (320), from
    before the score scalars six channels + six scalars (318), and from before
    the unseen channel five channels + six scalars (266). Works on numpy
    arrays and torch tensors."""
    if states.shape[-1] == obs_dim:
        return states
    if states.shape[-1] != OBS_DIM:
        raise ValueError(f"cannot adapt observation of width {states.shape[-1]} to a model expecting {obs_dim}")
    tail = OBS_DIM - SCALARS
    if obs_dim == CHANNELS_END + SCALARS:
        channels = CHANNELS_END
    elif obs_dim == CHANNELS_END + 6:
        channels, tail = CHANNELS_END, OBS_DIM - 6
    elif obs_dim == 5 * 52 + 6:
        channels, tail = 5 * 52, OBS_DIM - 6
    else:
        raise ValueError(f"cannot adapt observation of width {states.shape[-1]} to a model expecting {obs_dim}")
    if isinstance(states, np.ndarray):
        return np.concatenate([states[..., :channels], states[..., tail:]], axis=-1)
    return torch.cat([states[..., :channels], states[..., tail:]], dim=-1)


class VectorizedRummyEnv:
    """Steps all games inside the C++ engine in one call; no worker processes.

    With blank_known_prob > 0, a fraction of games hide the opponent-known-cards
    channel for their whole duration, so the policy learns to play with or
    without that information.
    """

    def __init__(self, num_envs, num_threads=0, seed=None, blank_known_prob=0.0, target_score=500, hand_size=7):
        if seed is None:
            seed = int(np.random.randint(0, 2**31))
        if num_threads <= 0:
            num_threads = min(num_envs, os.cpu_count() or 1)
        self.num_envs = num_envs
        self.blank_known_prob = blank_known_prob
        self.rng = np.random.default_rng(seed)
        self.blank = np.zeros(num_envs, dtype=bool)
        self._env = rummy_engine.VectorizedRummyEnv(num_envs, seed, num_threads, target_score, hand_size)

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
        """Returns (states, masks, rewards, dones, round_ends, round_outs); round_outs is
        the player who went out where a round ended, 0 otherwise."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        states, masks, rewards, dones, round_ends, round_outs = self._env.step(actions)
        if dones.any():
            self._resample_blank(dones)
        return (torch.from_numpy(self._apply_blank(states)), torch.from_numpy(masks),
                torch.from_numpy(rewards), torch.from_numpy(dones),
                torch.from_numpy(round_ends), torch.from_numpy(round_outs))

    def clone(self, i):
        """Independent copy of live game i, for search."""
        return self._env.get(int(i))

    def current_players(self):
        return torch.from_numpy(self._env.current_players())

    def opponent_hands(self):
        """Ground-truth cards of the player not to move, [N,52] bool. Training target only."""
        return torch.from_numpy(self._env.opponent_hands())

    def aux_targets(self):
        """Ground-truth auxiliary targets for the player to move, [N,AUX_DIM] float32. Training only."""
        return torch.from_numpy(self._env.aux_targets())

    def close(self):
        pass
