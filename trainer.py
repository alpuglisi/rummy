import copy
import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from env.vectorized_env import VectorizedRummyEnv
from models.ppo_network import AUX_GROUPS, RummyActorCritic, masked_categorical, sample_categorical
from config import PPOConfig
import glob

from env.vectorized_env import AUX_DIM, KNOWN_CARDS, OBS_DIM, UNSEEN_CARDS, adapt_obs, blank_known
from evaluate import DeckOnlyPolicy, GreedyPolicy, ModelPolicy, RandomPolicy, play_matches
from search import SearchPolicy

class RolloutBuffer:
    def __init__(self, cfg: PPOConfig, device: torch.device):
        self.states = torch.zeros((cfg.num_steps, cfg.num_envs, cfg.obs_dim), dtype=torch.float32).to(device)
        self.masks = torch.zeros((cfg.num_steps, cfg.num_envs, cfg.action_dim), dtype=torch.bool).to(device)
        self.actions = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.long).to(device)
        self.logprobs = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.float32).to(device)
        self.values = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.float32).to(device)
        self.dones = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.bool).to(device)
        # False where a pool opponent chose the action: excluded from the policy loss.
        self.valid = torch.ones((cfg.num_steps, cfg.num_envs), dtype=torch.bool).to(device)
        # Auxiliary targets the engine knows at the stored state, and labels
        # filled in later from what happened next (-1 = unknown, masked out).
        self.aux = torch.zeros((cfg.num_steps, cfg.num_envs, AUX_DIM), dtype=torch.float32).to(device)
        self.players = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.long).to(device)
        # The backfilled labels live on the host (they are written by
        # backfill_labels a few games at a time, which on the device would be a
        # handful of tiny kernels and uploads per finished game per step) and
        # are uploaded once per update, at the start of optimize. The player to
        # move at each state is kept on the host too, for the goes_out label.
        self.players_host = np.zeros((cfg.num_steps, cfg.num_envs), dtype=np.int32)
        self.next_discard = np.full((cfg.num_steps, cfg.num_envs), -1, dtype=np.int64)
        self.turns_left = np.full((cfg.num_steps, cfg.num_envs), -1.0, dtype=np.float32)
        self.goes_out = np.full((cfg.num_steps, cfg.num_envs), -1, dtype=np.int64)
        # Which auxiliary target groups are active for the game each state belongs to.
        self.aux_mask = torch.ones((cfg.num_steps, cfg.num_envs, len(AUX_GROUPS)), dtype=torch.float32).to(device)
        self.step = 0
        self.device = device

    def store(self, state, mask, action, logprob, reward, value, done, valid=None, aux=None, players=None,
              aux_mask=None, players_host=None):
        if valid is not None:
            self.valid[self.step] = valid.to(self.device)
        if aux is not None:
            self.aux[self.step] = aux.to(self.device)
        if aux_mask is not None:
            self.aux_mask[self.step] = aux_mask
        if players is not None:
            self.players[self.step] = players.to(self.device)
            if players_host is None:
                # The goes_out label reads the host history, so a caller that
                # only has the device players pays one sync here (the training
                # loop always passes its host copy).
                players_host = players.cpu().numpy()
        if players_host is not None:
            self.players_host[self.step] = players_host
        self.states[self.step] = state.to(self.device)
        self.masks[self.step] = mask.to(self.device)
        self.actions[self.step] = action.to(self.device)
        self.logprobs[self.step] = logprob.to(self.device)
        self.rewards[self.step] = reward.to(self.device)
        self.values[self.step] = value.squeeze(-1).to(self.device)
        self.dones[self.step] = done.to(self.device)
        self.step += 1

    def compute_advantages(self, next_value, next_state, next_done, gamma=0.99, gae_lambda=0.95):
        advantages = torch.zeros_like(self.rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.step)):
            if t == self.step - 1:
                nextnonterminal = 1.0 - next_done.float()
                nextvalues = next_value
                next_is_same_player = (next_state[..., -3] == 1.0).float()
            else:
                nextnonterminal = 1.0 - self.dones[t + 1].float()
                nextvalues = self.values[t + 1]
                next_is_same_player = (self.states[t + 1][..., -3] == 1.0).float()
                
            perspective_coef = 2.0 * next_is_same_player - 1.0
            
            delta = self.rewards[t] + gamma * nextvalues * perspective_coef * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * perspective_coef * nextnonterminal * lastgaelam
            
        returns = advantages + self.values
        return advantages, returns

    def clear(self):
        self.step = 0
        self.next_discard.fill(-1)
        self.turns_left.fill(-1.0)
        self.goes_out.fill(-1)


class GraphedPolicy:
    """A frozen pool member's forward replayed from captured CUDA graphs.

    A member forward at batch 32-128 is ~160 kernel launches for about a
    millisecond of GPU work, so it is launch-bound; a graph replay is one
    launch. One graph is captured per batch-size bucket (powers of two from
    32 up to num_envs) on first use, with static input and output tensors.
    act() copies the k live rows into the static inputs, pads the remaining
    rows by repeating row 0 (the network treats every row independently:
    LayerNorm, attention and the card mean all stay inside the row), replays,
    and returns the first k rows of the static logits. The caller samples
    from them at once, before any other graph replays.

    All graphs of all members capture from one shared memory pool. That is
    safe because every graph's output is consumed (sampled into the action
    tensor on the same stream) before any other graph replays, and the static
    outputs themselves stay referenced, so no later capture can reuse their
    memory. Any exception or a replay that disagrees with an eager forward
    switches the member to eager forwards for good.
    """

    _pool_handle = None

    def __init__(self, model, num_envs):
        self.model = model
        self.graphs = {}
        self.ok = True
        self.buckets = []
        bucket = 32
        while bucket < num_envs:
            self.buckets.append(bucket)
            bucket *= 2
        self.buckets.append(bucket)

    @classmethod
    def pool_handle(cls):
        if cls._pool_handle is None:
            cls._pool_handle = torch.cuda.graph_pool_handle()
        return cls._pool_handle

    @torch.no_grad()
    def _capture(self, bucket):
        device = next(self.model.parameters()).device
        # Realistic inputs for the warm-up and the self-check, from a private
        # generator so the global RNG stream stays untouched.
        gen = torch.Generator().manual_seed(bucket)
        static_state = (torch.rand(bucket, self.model.obs_dim, generator=gen) < 0.2).float().to(device)
        static_mask = (torch.rand(bucket, 105, generator=gen) < 0.5).to(device)
        static_mask[:, 0] = True
        # Capture protocol from the torch.cuda.graph docs: warm up on a side
        # stream (lazy initialisation of cuBLAS/cuDNN must not land in the
        # graph), then capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.model(static_state, static_mask)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool_handle()):
            static_logits, _ = self.model(static_state, static_mask)
        graph.replay()
        reference, _ = self.model(static_state, static_mask)
        if not torch.allclose(static_logits, reference, atol=1e-4, rtol=1e-3):
            raise RuntimeError("replayed logits differ from the eager forward")
        self.graphs[bucket] = (graph, static_state, static_mask, static_logits)
        return self.graphs[bucket]

    def act(self, state, mask):
        """Masked logits for the rows of `state` / `mask` (already adapted to the member's obs_dim)."""
        k = state.shape[0]
        if self.ok:
            bucket = next(b for b in self.buckets if b >= k)
            # A capture that fails inside torch.cuda.graph leaves its side
            # stream current (the context manager's __exit__ raises from
            # capture_end before restoring the stream), so remember the
            # caller's stream to put it back on the eager fallback.
            prev_stream = torch.cuda.current_stream()
            try:
                graph, static_state, static_mask, static_logits = self.graphs.get(bucket) or self._capture(bucket)
                static_state[:k].copy_(state)
                static_mask[:k].copy_(mask)
                if k < bucket:
                    static_state[k:].copy_(state[:1].expand(bucket - k, -1))
                    static_mask[k:].copy_(mask[:1].expand(bucket - k, -1))
                graph.replay()
                return static_logits[:k]
            except Exception as e:
                print(f"Warning: CUDA graph for a pool member failed at batch {bucket} "
                      f"({type(e).__name__}: {e}); that member runs eagerly from now on.")
                self.ok = False
                self.graphs.clear()
                torch.cuda.set_stream(prev_stream)
                torch.cuda.synchronize()
        logits, _ = self.model(state, mask)
        return logits


class PPOTrainer:
    def __init__(self, config: PPOConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        # TF32 matmuls: free speed on Ampere+ for these linear layers at no cost to PPO.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        if config.obs_dim != OBS_DIM:
            raise ValueError(f"config.obs_dim is {config.obs_dim} but the engine observation is {OBS_DIM} wide")
        self.envs = VectorizedRummyEnv(config.num_envs, config.env_threads,
                                       blank_known_prob=config.blank_known_prob, hand_size=config.hand_size)
        if config.init_checkpoint:
            self.model = RummyActorCritic.from_state_dict(
                torch.load(config.init_checkpoint, map_location=self.device)).to(self.device)
            if self.model.obs_dim != config.obs_dim:
                raise ValueError(f"init_checkpoint '{config.init_checkpoint}' expects an observation width of "
                                 f"{self.model.obs_dim}, but the current engine produces {config.obs_dim}; "
                                 f"it was saved under an older observation layout and cannot be used to warm-start "
                                 f"the live policy (adapt_obs is only for opponents, not the model being trained)")
            print(f"Warm-started the live policy from {config.init_checkpoint} (arch={self.model.arch}); "
                 f"config.arch/hidden_size/token_dim/token_layers are ignored for it. The optimizer, opponent "
                 f"pool and learning-rate schedule all start fresh.")
        else:
            self.model = RummyActorCritic(config.obs_dim, config.action_dim, config.hidden_size,
                                          config.num_layers, config.residual, arch=config.arch,
                                          token_dim=config.token_dim, token_layers=config.token_layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.buffer = RolloutBuffer(config, self.device)

        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.frozen_model = None
        self.best_model = None      # incumbent best policy (see config.best_checkpoint)
        self.best_step = 0
        self.evals_run = 0
        self.rng = np.random.default_rng()
        # Separate generator for the per-game auxiliary dropout, seeded from OS
        # entropy so every game's choice is independent of everything else.
        self.aux_rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
        self.aux_mask_stage = self.staging((config.num_envs, len(AUX_GROUPS)), torch.float32)
        self.aux_mask_host = self.aux_mask_stage.numpy()
        self.aux_mask_host[:] = 1.0
        self.aux_mask = torch.ones((config.num_envs, len(AUX_GROUPS)), dtype=torch.float32, device=self.device)
        self.last_aux = None

        # Host staging for everything the step uploads, page-locked when the
        # device is CUDA so every upload is asynchronous and the host never
        # waits for the GPU except at the one action readback per step. The
        # engine writes straight into the numpy views. Reuse safety: a pinned
        # buffer may be overwritten only after a stream synchronisation that
        # follows its previous upload. The per-step action.cpu() is that
        # synchronisation, so a buffer written after it in step t (state,
        # mask, reward, done, players, and the opponent / aux-dropout
        # bookkeeping) was last uploaded in step t-1, before the sync of step
        # t. The buffers written before the sync in step t (the aux targets,
        # the pool index segments and the learner-turn mask) were last written
        # and uploaded in step t-1, also before that step's sync. Every buffer
        # therefore needs a single copy.
        n = config.num_envs
        self.state_stage = self.staging((n, config.obs_dim), torch.float32)
        self.mask_stage = self.staging((n, config.action_dim), torch.bool)
        self.reward_stage = self.staging((n,), torch.float32)
        self.done_stage = self.staging((n,), torch.bool)
        self.players_stage = self.staging((n,), torch.long)
        self.aux_stage = self.staging((n, AUX_DIM), torch.float32)
        self.valid_stage = self.staging((n,), torch.bool)
        self.pool_idx_stage = self.staging((n,), torch.long)
        self.state_host = self.state_stage.numpy()
        self.mask_host = self.mask_stage.numpy()
        self.reward_host = self.reward_stage.numpy()
        self.done_host = self.done_stage.numpy()
        self.players_host = self.players_stage.numpy()
        self.aux_host = self.aux_stage.numpy()
        self.valid_host = self.valid_stage.numpy()
        self.pool_idx_host = self.pool_idx_stage.numpy()
        self.round_ends_host = np.zeros(n, dtype=bool)
        self.round_outs_host = np.zeros(n, dtype=np.int32)
        # Engines built with the *_into calls fill the staging directly; older
        # builds return fresh arrays that are copied over.
        self.engine_into = hasattr(self.envs._env, "step_into") and hasattr(self.envs._env, "aux_targets_into")
        self.teacher = SearchPolicy(self.model, self.device, worlds=config.distill_worlds,
                                    max_actions=config.distill_actions,
                                    seed=int(self.rng.integers(0, 2**31)),
                                    horizon=config.distill_horizon, endgame=config.search_endgame,
                                    reward_scale=config.reward_scale, replies=config.distill_replies,
                                    max_sims=config.search_max_sims)

        # Opponent pool: frozen policies; each game is either pure self-play
        # (opponent -1) or has one pool member controlling one seat. Members'
        # forwards replay from CUDA graphs when the device allows (pool_graphs
        # is kept aligned with pool; None means an eager forward).
        self.pool = []
        self.pool_graphs = []
        self.use_graphs = self.device.type == "cuda" and config.pool_cuda_graphs
        if config.pool_fraction > 0 and config.pool_init_dir and os.path.isdir(config.pool_init_dir):
            for path in sorted(glob.glob(os.path.join(config.pool_init_dir, "*.pth")))[-config.pool_size:]:
                model = RummyActorCritic.from_state_dict(torch.load(path, map_location=self.device))
                self.pool.append(model.to(self.device).eval())
                self.pool_graphs.append(GraphedPolicy(model, config.num_envs) if self.use_graphs else None)
            if self.pool:
                print(f"Opponent pool seeded with {len(self.pool)} checkpoint(s) from {config.pool_init_dir}/")
        self.pool_wins = [0.0] * len(self.pool)
        self.pool_games = [0.0] * len(self.pool)
        # pool_wins / pool_games: decayed win and game counts of the learner
        # against each member, for PFSP sampling (kept aligned with self.pool).
        # Per-game bookkeeping (opponent, seat, round start, dropout draws)
        # lives on the host: it is tiny, and keeping it there means no step
        # has to wait for the GPU to read it back. Device copies are kept for
        # the tensors the losses need.
        self.opponent_stage = self.staging((config.num_envs,), torch.long)
        self.opponent_seat_stage = self.staging((config.num_envs,), torch.int32)
        self.opponent_host = self.opponent_stage.numpy()
        self.opponent_seat_host = self.opponent_seat_stage.numpy()
        self.opponent_host[:] = -1
        self.opponent_seat_host[:] = 1
        self.opponent = torch.full((config.num_envs,), -1, dtype=torch.long, device=self.device)
        self.opponent_seat = torch.ones(config.num_envs, dtype=torch.int32, device=self.device)
        self.round_start = np.zeros(config.num_envs, dtype=np.int64)

        # The minibatch losses: one Python function, run compiled when that
        # works (and is wanted) and eagerly otherwise, so the two cannot
        # diverge in what they compute.
        self.losses = self.compile_losses() if config.compile_optimize else self.minibatch_losses

    def staging(self, shape, dtype):
        """Host tensor the engine writes and the device reads: page-locked on
        CUDA so the upload can be asynchronous, a plain tensor elsewhere."""
        return torch.empty(shape, dtype=dtype, pin_memory=self.device.type == "cuda")

    def resample_aux_mask(self, which):
        """Draw a fresh set of active auxiliary target groups for the games in `which`
        (a host bool array; each group is dropped for the whole game with
        probability aux_dropout)."""
        envs = np.flatnonzero(which)
        if len(envs) == 0 or self.cfg.aux_dropout <= 0:
            return
        keep = self.aux_rng.random((len(envs), len(AUX_GROUPS))) >= self.cfg.aux_dropout
        self.aux_mask_host[envs] = keep
        self.aux_mask.copy_(self.aux_mask_stage, non_blocking=True)

    def pool_win_rates(self):
        """Learner's decayed win rate against each pool member (0.5 until it has played them)."""
        return [w / g if g >= 1.0 else 0.5 for w, g in zip(self.pool_wins, self.pool_games)]

    def assign_opponents(self, which):
        """Pick a pool opponent (or none) and a seat for the games in `which` (a host bool array)."""
        envs = np.flatnonzero(which)
        n = len(envs)
        if n == 0:
            return
        if not self.pool or self.cfg.pool_fraction <= 0:
            self.opponent_host[envs] = -1
        else:
            use_pool = self.rng.random(n) < self.cfg.pool_fraction
            if self.cfg.pool_pfsp:
                # Prioritised fictitious self-play: favour the members the learner loses to.
                rates = np.array(self.pool_win_rates())
                weights = np.clip(1.0 - rates, 0.0, None) ** self.cfg.pool_pfsp_power + 0.05
                ids = self.rng.choice(len(self.pool), size=n, p=weights / weights.sum())
            else:
                ids = self.rng.integers(len(self.pool), size=n)
            self.opponent_host[envs] = np.where(use_pool, ids, -1)
            self.opponent_seat_host[envs] = self.rng.integers(1, 3, size=n)
        self.opponent.copy_(self.opponent_stage, non_blocking=True)
        self.opponent_seat.copy_(self.opponent_seat_stage, non_blocking=True)

    def update_pool_stats(self, reward, players, done, decay=0.995):
        """Record who won each finished pool game (from the terminal reward's sign).
        All three arguments are host arrays."""
        finished = np.flatnonzero(done & (self.opponent_host >= 0))
        for e in finished:
            actor_is_pool = players[e] == self.opponent_seat_host[e]
            r = reward[e]
            won = 0.5 if r == 0 else float((r > 0) != actor_is_pool)
            pid = int(self.opponent_host[e])
            self.pool_games[pid] = self.pool_games[pid] * decay + 1.0
            self.pool_wins[pid] = self.pool_wins[pid] * decay + won

    def add_pool_member(self, model):
        self.pool.append(model)
        self.pool_graphs.append(GraphedPolicy(model, self.cfg.num_envs) if self.use_graphs else None)
        self.pool_wins.append(0.0)
        self.pool_games.append(0.0)
        if len(self.pool) > self.cfg.pool_size:
            self.pool.pop(0)
            self.pool_graphs.pop(0)   # its captured graphs go with it
            self.pool_wins.pop(0)
            self.pool_games.pop(0)
            # Games already running hold the index of their opponent, so the
            # shift has to be applied to them too: without it every game keeps
            # its old number and silently starts playing the next member up,
            # and its result is then credited to that member's PFSP record.
            # Games whose opponent was the one evicted fall back to self-play.
            evicted = self.opponent_host == 0
            self.opponent_host[self.opponent_host > 0] -= 1
            self.opponent_host[evicted] = -1
            self.opponent.copy_(self.opponent_stage, non_blocking=True)

    def backfill_labels(self, t, action, players, round_ends, round_outs, done):
        """Fill in labels that depend on what happens next.

        A turn is two consecutive buffer steps of the same game (draw, then
        discard), so when a player discards at step t the opponent's last
        turn was steps t-2 and t-3: those states get 'opponent's next discard'.
        When a round ends, every state since it started gets the turns that
        were left and who went out (me / opponent / nobody). Everything here
        is host arrays: the labels are uploaded once per update."""
        buf = self.buffer
        discards = action >= 53
        for k in (2, 3):
            if t - k < 0:
                break
            envs = np.flatnonzero(discards & (self.round_start <= t - k))
            if len(envs):
                buf.next_discard[t - k, envs] = action[envs] - 53
        for e in np.flatnonzero(round_ends | done):
            i0 = int(self.round_start[e])
            left = (t - np.arange(i0, t + 1)).astype(np.float32)
            left /= 2.0
            left /= 40.0
            buf.turns_left[i0:t + 1, e] = left
            out = int(round_outs[e])
            if out == 0:
                buf.goes_out[i0:t + 1, e] = 2
            else:
                buf.goes_out[i0:t + 1, e] = np.where(buf.players_host[i0:t + 1, e] == out, 0, 1)
            self.round_start[e] = t + 1

    @torch.no_grad()
    def pool_actions(self, action, state, mask, players, players_host):
        """Override the learner's actions where a pool opponent is to move; return the learner-turn mask.

        Which games each member controls is worked out from the host copies, so
        the GPU never has to be read back mid-step. All members' row indices go
        up in one upload as contiguous segments of one staging buffer (and the
        learner-turn mask in another); each member's forward stays its own."""
        turn_host = (self.opponent_host >= 0) & (players_host == self.opponent_seat_host)
        np.logical_not(turn_host, out=self.valid_host)
        segments = []
        end = 0
        for pid in np.unique(self.opponent_host[turn_host]):
            rows = np.flatnonzero(turn_host & (self.opponent_host == pid))
            self.pool_idx_host[end:end + len(rows)] = rows
            segments.append((int(pid), end, end + len(rows)))
            end += len(rows)
        valid = self.valid_stage.to(self.device, non_blocking=True, copy=True)
        if segments:
            idx_all = self.pool_idx_stage[:end].to(self.device, non_blocking=True, copy=True)
            for pid, start, stop in segments:
                idx = idx_all[start:stop]
                model, graphed = self.pool[pid], self.pool_graphs[pid]
                obs, rows_mask = adapt_obs(state[idx], model.obs_dim), mask[idx]
                logits = graphed.act(obs, rows_mask) if graphed is not None else model(obs, rows_mask)[0]
                action[idx] = sample_categorical(masked_categorical(logits))
        return valid

    def anneal_lr(self, global_step):
        cfg = self.cfg
        progress = global_step / cfg.total_timesteps
        frac = min(1.0, max(0.0, (progress - cfg.lr_anneal_start) / max(1e-9, 1.0 - cfg.lr_anneal_start)))
        lr = cfg.learning_rate + (cfg.lr_final - cfg.learning_rate) * frac
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr
        
    def engine_step(self, action_host):
        """Step the games and leave the new observation in the staging buffers.

        The same as VectorizedRummyEnv.step (including the blank-known
        resample for finished games), writing into the staging instead of
        fresh arrays where the engine can."""
        envs = self.envs
        if self.engine_into:
            envs._env.step_into(action_host, self.state_host, self.mask_host, self.reward_host, self.done_host,
                                self.round_ends_host, self.round_outs_host)
            if self.done_host.any():
                envs._resample_blank(self.done_host)
            envs._apply_blank(self.state_host)
        else:
            state, mask, reward, done, round_ends, round_outs = envs.step(action_host)
            np.copyto(self.state_host, state.numpy())
            np.copyto(self.mask_host, mask.numpy())
            np.copyto(self.reward_host, reward.numpy())
            np.copyto(self.done_host, done.numpy())
            np.copyto(self.round_ends_host, round_ends.numpy())
            np.copyto(self.round_outs_host, round_outs.numpy())

    def aux_targets(self):
        """Auxiliary targets of the games' current states, uploaded from the staging."""
        if self.engine_into:
            self.envs._env.aux_targets_into(self.aux_host)
        else:
            np.copyto(self.aux_host, self.envs.aux_targets().numpy())
        return self.aux_stage.to(self.device, non_blocking=True, copy=True)

    def train(self):
        state, mask = self.envs.reset()
        state = state.to(self.device)
        mask = mask.to(self.device)
        done = torch.zeros(self.cfg.num_envs, dtype=torch.bool).to(self.device)
        np.copyto(self.players_host, self.envs.current_players().numpy())
        players = self.players_stage.to(self.device, non_blocking=True, copy=True)
        everyone = np.ones(self.cfg.num_envs, dtype=bool)
        self.assign_opponents(everyone)
        self.resample_aux_mask(everyone)

        global_step = 0
        update = 0
        while global_step < self.cfg.total_timesteps:
            t_start = time.time()
            lr = self.anneal_lr(global_step)
            learner_steps = torch.zeros((), device=self.device)
            self.round_start[:] = 0
            for t in range(self.cfg.num_steps):
                with torch.no_grad():
                    action, logprob, value = self.model.get_action(state, mask)
                valid = self.pool_actions(action, state, mask, players, self.players_host)
                learner_steps += valid.sum()
                # The engine's targets for the current state, computed while
                # the GPU works through the forwards just launched (the games
                # do not move until engine_step below).
                aux_targets = self.aux_targets()

                # The one host read per step: the engine needs the actions.
                action_host = action.cpu().numpy()
                self.engine_step(action_host)
                self.reward_host *= self.cfg.reward_scale
                reward = self.reward_stage.to(self.device, non_blocking=True, copy=True)
                next_done = self.done_stage.to(self.device, non_blocking=True, copy=True)

                self.buffer.store(state, mask, action, logprob, reward, value, done, valid, aux_targets, players,
                                  self.aux_mask, self.players_host)
                self.backfill_labels(t, action_host, self.players_host, self.round_ends_host, self.round_outs_host,
                                     self.done_host)
                self.update_pool_stats(self.reward_host, self.players_host, self.done_host)

                state = self.state_stage.to(self.device, non_blocking=True, copy=True)
                mask = self.mask_stage.to(self.device, non_blocking=True, copy=True)
                done = next_done
                np.copyto(self.players_host, self.envs.current_players().numpy())
                players = self.players_stage.to(self.device, non_blocking=True, copy=True)
                if self.done_host.any():
                    self.assign_opponents(self.done_host)
                    self.resample_aux_mask(self.done_host)
                global_step += self.cfg.num_envs

            t_rollout = time.time()
            with torch.no_grad():
                _, next_value = self.model(state, mask)
                next_value = next_value.squeeze(-1)

            advantages, returns = self.buffer.compute_advantages(
                next_value, state, done, self.cfg.gamma, self.cfg.gae_lambda
            )

            distill = None
            if self.cfg.distill_coef > 0 and update % self.cfg.distill_every == 0:
                distill = self.collect_search_targets()
                self.writer.add_scalar("Search/TeacherAgreement", distill["agreement"], global_step)
                self.writer.add_scalar("Search/RolloutSteps", distill["rollout_steps"], global_step)
                self.writer.add_scalar("Perf/DistillSec", distill["seconds"], global_step)
                self.writer.add_scalar("Search/TeacherBelief", float(self.teacher.belief), global_step)
            t_distill = time.time()

            actor_loss, critic_loss, distill_loss, aux, policy = self.optimize(advantages, returns, distill)
            self.last_aux = aux
            t_optimize = time.time()

            self.writer.add_scalar("Loss/Actor", actor_loss, global_step)
            self.writer.add_scalar("Loss/Critic", critic_loss, global_step)
            if distill is not None:
                self.writer.add_scalar("Loss/Distill", distill_loss, global_step)
            self.writer.add_scalar("Train/LearningRate", lr, global_step)
            for name, value in policy.items():
                self.writer.add_scalar(f"Policy/{name}", value, global_step)
            if self.cfg.aux_coef > 0:
                for name, value in aux.items():
                    if name not in ("precision", "baseline"):
                        self.writer.add_scalar(f"Aux/{name}", value, global_step)
                self.writer.add_scalar("Aux/TopKPrecision", aux["precision"], global_step)
                self.writer.add_scalar("Aux/TopKBaseline", aux["baseline"], global_step)
            self.writer.add_scalar("Pool/Size", len(self.pool), global_step)
            if self.pool:
                rates = [r for r, g in zip(self.pool_win_rates(), self.pool_games) if g >= 5]
                if rates:
                    self.writer.add_scalar("Pool/HardestWinRate", min(rates), global_step)
                    self.writer.add_scalar("Pool/MeanWinRate", sum(rates) / len(rates), global_step)
            self.writer.add_scalar("Pool/LearnerStepFraction",
                                   float(learner_steps) / (self.cfg.num_steps * self.cfg.num_envs), global_step)
            self.writer.add_scalar("Reward/Average_Return", returns.mean().item(), global_step)

            self.buffer.clear()
            samples = self.cfg.num_envs * self.cfg.num_steps
            train_sps = samples / (t_optimize - t_start)
            print(f"Global Step: {global_step} / {self.cfg.total_timesteps}  ({train_sps:,.0f} steps/s)")

            update += 1
            if self.cfg.pool_fraction > 0 and update % self.cfg.pool_add_every == 0:
                self.add_pool_member(copy.deepcopy(self.model).eval())
            if update % self.cfg.eval_interval == 0:
                self.evaluate(global_step)
            t_end = time.time()

            self.writer.add_scalar("Perf/RolloutSec", t_rollout - t_start, global_step)
            self.writer.add_scalar("Perf/OptimizeSec", t_optimize - t_distill, global_step)
            self.writer.add_scalar("Perf/EvalSec", t_end - t_optimize, global_step)
            self.writer.add_scalar("Perf/TrainStepsPerSec", train_sps, global_step)
            self.writer.add_scalar("Perf/StepsPerSec", samples / (t_end - t_start), global_step)
            
            current_millions = global_step // 1_000_000

            
            previous_millions = (global_step - (self.cfg.num_envs * self.cfg.num_steps)) // 1_000_000

            
            if current_millions > previous_millions:

            
                self.save_checkpoint(f"checkpoints/ppo_rummy_{global_step}.pth")

            
                print(f"Checkpoint saved at step {global_step}!")

    def collect_search_targets(self):
        """Run the search on a sample of live training positions and turn the
        per-action outcomes into a target distribution for the policy."""
        cfg = self.cfg
        t0 = time.time()
        idx = self.rng.choice(cfg.num_envs, size=min(cfg.distill_positions, cfg.num_envs), replace=False)
        envs = [self.envs.clone(i) for i in idx]

        self.model.eval()
        # Belief-weighted worlds only once the opponent-hand head beats random guessing
        # by a margin; a weak belief makes the worlds wrong in a consistent direction.
        gain = self.last_aux["precision"] - self.last_aux["baseline"] if self.last_aux else 0.0
        self.teacher.belief = bool(cfg.distill_belief and cfg.aux_coef > 0
                                   and gain >= cfg.distill_belief_min_gain)
        self.teacher.reset_stats()
        stats = self.teacher.evaluate_actions(envs)

        # The policy learns from its own view of each position: hide the
        # opponent-known channel where that training game has it hidden.
        obs = np.stack([e.get_state() for e in envs])
        masks = np.stack([e.get_legal_actions() for e in envs]).astype(bool)
        blank = self.envs.blank[idx]
        if blank.any():
            obs[blank] = blank_known(obs[blank])

        targets = np.zeros((len(envs), cfg.action_dim), dtype=np.float32)
        for j, s in enumerate(stats):
            acts = np.fromiter(s.keys(), dtype=np.int64)
            vals = np.array([s[a][0] for a in acts])
            w = np.exp((vals - vals.max()) / cfg.distill_temperature)
            targets[j, acts] = w / w.sum()

        obs_t = torch.as_tensor(obs, device=self.device)
        mask_t = torch.as_tensor(masks, device=self.device)
        with torch.no_grad():
            model_pick = self.model(obs_t, mask_t)[0].argmax(dim=-1).cpu().numpy()
        self.model.train()

        return {
            "states": obs_t,
            "masks": mask_t,
            "targets": torch.as_tensor(targets, device=self.device),
            "agreement": float((model_pick == targets.argmax(axis=1)).mean()),
            "rollout_steps": self.teacher.stats()["mean_rollout_steps"],
            "seconds": time.time() - t0,
        }

    def aux_losses(self, out, states, targets, next_discard, turns_left, goes_out, active=None):
        """Auxiliary losses for one minibatch. Returns (weighted sum, per-target stats).

        `targets` is the engine's AUX block for each state (see rummy_env.h);
        the next three are labels backfilled from later steps (-1 = unknown);
        `active` [B, len(AUX_GROUPS)] says which target groups count for each
        sample's game (per-game dropout). Every loss is a per-sample quantity
        averaged over the samples where its group is active and its label known."""
        w = self.cfg.aux_weights
        hand = states[:, :52]
        stats = {}
        per_sample = {}

        def masked_mean(values, valid):
            return (values * valid).sum() / valid.sum().clamp(min=1.0)

        opp = targets[:, 0:52]
        per_sample["opponent"] = F.binary_cross_entropy_with_logits(out["opponent"], opp, reduction="none").mean(1)
        per_sample["layoff"] = F.binary_cross_entropy_with_logits(out["layoff"], targets[:, 52:104],
                                                                  reduction="none").mean(1)
        # Only the player's own hand cards can be discarded: average over those.
        n_hand = hand.sum(1).clamp(min=1.0)
        takeable = F.binary_cross_entropy_with_logits(out["takeable"], targets[:, 104:156], reduction="none")
        per_sample["takeable"] = (takeable * hand).sum(1) / n_hand
        value_err = (out["discard_value"] - targets[:, 156:208]) ** 2
        per_sample["discard_value"] = (value_err * hand).sum(1) / n_hand
        flags = targets[:, [208, 210, 211]]   # opponent holds a meld, can go out next turn, deck completes a meld
        per_sample["flags"] = F.binary_cross_entropy_with_logits(out["flags"], flags, reduction="none").mean(1)
        per_sample["hand_points"] = (out["hand_points"].squeeze(-1) - targets[:, 209]) ** 2
        known = {}
        if "next_discard" in out:
            per_sample["next_discard"] = F.cross_entropy(out["next_discard"], next_discard, ignore_index=-1,
                                                         reduction="none")
            known["next_discard"] = (next_discard >= 0).float()
            per_sample["turns_left"] = (out["turns_left"].squeeze(-1) - turns_left) ** 2
            known["turns_left"] = (turns_left >= 0).float()
            per_sample["goes_out"] = F.cross_entropy(out["goes_out"], goes_out, ignore_index=-1, reduction="none")
            known["goes_out"] = (goes_out >= 0).float()

        losses = {}
        for i, name in enumerate(AUX_GROUPS):
            if name not in per_sample:
                continue
            valid = known.get(name, torch.ones_like(per_sample[name]))
            if active is not None:
                valid = valid * active[:, i]
            losses[name] = masked_mean(per_sample[name], valid)
        total = sum(w.get(name, 1.0) * l for name, l in losses.items())

        with torch.no_grad():
            for name, l in losses.items():
                stats[f"Loss_{name}"] = l.detach()
            stats["TotalLoss"] = total.detach()
            # Precision of the top-k predicted cards, k = opponent hand size, against a
            # random guess among the cards that could be in their hand.
            k = opp.sum(dim=1)
            top = torch.topk(out["opponent"], 26, dim=1).indices
            rank_ok = (torch.arange(26, device=states.device)[None, :] < k[:, None]).float()
            candidates = states[:, KNOWN_CARDS].sum(dim=1) + states[:, UNSEEN_CARDS].sum(dim=1)
            stats["_hits"] = (opp.gather(1, top) * rank_ok).sum()
            stats["_total"] = k.sum()
            stats["_baseline"] = (k * k / candidates.clamp(min=1)).sum()
            if "next_discard" in out:
                m = next_discard >= 0
                stats["NextDiscardAcc"] = ((out["next_discard"].argmax(1) == next_discard) & m).sum() / m.sum().clamp(min=1)
                m = goes_out >= 0
                stats["WhoGoesOutAcc"] = ((out["goes_out"].argmax(1) == goes_out) & m).sum() / m.sum().clamp(min=1)
                m = turns_left >= 0
                stats["TurnsLeftMAE"] = ((out["turns_left"].squeeze(-1) - turns_left).abs() * m).sum() / m.sum().clamp(min=1) * 40
                stats["CanGoOutAcc"] = ((out["flags"][:, 1] > 0).float() == flags[:, 1]).float().mean()
                cards = hand.sum().clamp(min=1.0)
                stats["TakeableAcc"] = (((out["takeable"] > 0).float() == targets[:, 104:156]).float() * hand).sum() / cards
                stats["DiscardValueMAE"] = ((out["discard_value"] - targets[:, 156:208]).abs() * hand).sum() / cards * 50
        return total, stats

    def minibatch_losses(self, mb_states, mb_masks, mb_actions, mb_old_logprobs, mb_advantages, mb_returns,
                         mb_valid, mb_aux, mb_next, mb_turns, mb_out, mb_amask,
                         d_states=None, d_masks=None, d_targets=None):
        """Forward and losses of one minibatch, as a pure function of tensors so
        it can be compiled. Returns (loss, actor, critic, distill, aux stats,
        logits finite, policy stats); everything after `loss` is detached and
        distill is None without distillation inputs. The backward and the
        optimiser step stay outside."""
        cfg = self.cfg
        n_valid = mb_valid.sum().clamp(min=1.0)

        logits, new_values, aux_out = self.model.forward_with_aux(mb_states, mb_masks)
        # The validated Categorical used to be the only guard against broken
        # logits; its checks sync the GPU, so the guard is a device flag instead.
        finite = torch.isfinite(logits).all()
        dist = masked_categorical(logits)

        new_logprobs = dist.log_prob(mb_actions)
        entropy = (dist.entropy() * mb_valid).sum() / n_valid

        logratio = new_logprobs - mb_old_logprobs
        ratio = logratio.exp()

        pg_loss1 = mb_advantages * ratio
        pg_loss2 = mb_advantages * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
        actor_loss = -(torch.min(pg_loss1, pg_loss2) * mb_valid).sum() / n_valid

        critic_loss = F.mse_loss(new_values.squeeze(-1), mb_returns)

        loss = actor_loss + (cfg.vf_coef * critic_loss) - (cfg.ent_coef * entropy)

        # Diagnostics over the learner's own moves only, same masking as the
        # losses: entropy says whether the policy has gone deterministic, and
        # the clip fraction / approximate KL say whether each update is too
        # aggressive (pinned at the clip boundary) or too timid (~0).
        with torch.no_grad():
            clipped = ((ratio - 1.0).abs() > cfg.clip_coef).float()
            pstats = {
                "entropy": entropy.detach(),
                "clip_fraction": (clipped * mb_valid).sum() / n_valid,
                "approx_kl": (((ratio - 1.0) - logratio) * mb_valid).sum() / n_valid,
            }

        stats = {}
        if cfg.aux_coef > 0:
            aux_loss, stats = self.aux_losses(aux_out, mb_states, mb_aux, mb_next, mb_turns, mb_out, mb_amask)
            loss = loss + cfg.aux_coef * aux_loss

        distill_loss = None
        if d_states is not None:
            d_logits, _ = self.model(d_states, d_masks)
            # Illegal actions carry -inf log-probs and zero target mass; clamp keeps 0 * -inf out.
            d_logp = F.log_softmax(d_logits, dim=-1).clamp(min=-1e4)
            distill_loss = -(d_targets * d_logp).sum(dim=-1).mean()
            loss = loss + cfg.distill_coef * distill_loss
            distill_loss = distill_loss.detach()

        return loss, actor_loss.detach(), critic_loss.detach(), distill_loss, stats, finite, pstats

    def compile_losses(self):
        """torch.compile minibatch_losses (the ~700 small kernels of a minibatch
        become a few fused ones) and warm it up on random inputs of the real
        shapes, once per variant (with and without distillation inputs), so
        no compile lands inside a training update. Any failure means the
        eager function is used instead."""
        cfg = self.cfg
        fn = torch.compile(self.minibatch_losses, dynamic=False)
        gen = torch.Generator().manual_seed(0)   # private: the global RNG stream stays untouched

        def rand(*shape):
            return torch.rand(*shape, generator=gen).to(self.device)

        b = min(cfg.batch_size, cfg.num_envs * cfg.num_steps)
        g = len(AUX_GROUPS)
        args = [(rand(b, cfg.obs_dim) < 0.2).float(), rand(b, cfg.action_dim) < 0.5,
                (rand(b) * cfg.action_dim).long(), -rand(b), rand(b) - 0.5, rand(b) - 0.5,
                (rand(b) < 0.5).float(), (rand(b, AUX_DIM) < 0.5).float(), (rand(b) * 53).long() - 1,
                rand(b) - 0.1, (rand(b) * 4).long() - 1, (rand(b, g) < 0.5).float()]
        args[1][:, 0] = True
        variants = [()]
        if cfg.distill_coef > 0:
            d = min(cfg.batch_size, cfg.distill_positions, cfg.num_envs)
            d_masks = rand(d, cfg.action_dim) < 0.5
            d_masks[:, 0] = True
            variants.append(((rand(d, cfg.obs_dim) < 0.2).float(), d_masks,
                             torch.softmax(rand(d, cfg.action_dim), dim=-1)))
        try:
            for extra in variants:
                loss = fn(*args, *extra)[0]
                loss.backward()
        except Exception as e:
            print(f"Warning: torch.compile of the minibatch losses failed ({type(e).__name__}: {e}); "
                  f"the optimiser runs eagerly.")
            fn = self.minibatch_losses
        self.optimizer.zero_grad()
        return fn

    def optimize(self, advantages, returns, distill=None):
        b_states = self.buffer.states.view(-1, self.cfg.obs_dim)
        b_masks = self.buffer.masks.view(-1, self.cfg.action_dim)
        b_actions = self.buffer.actions.view(-1)
        b_logprobs = self.buffer.logprobs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_valid = self.buffer.valid.view(-1).float()
        b_aux = self.buffer.aux.view(-1, AUX_DIM)
        # The labels were backfilled on the host during the rollout: one upload each.
        b_next = torch.from_numpy(self.buffer.next_discard).to(self.device).view(-1)
        b_turns = torch.from_numpy(self.buffer.turns_left).to(self.device).view(-1)
        b_out = torch.from_numpy(self.buffer.goes_out).to(self.device).view(-1)
        b_amask = self.buffer.aux_mask.view(-1, len(AUX_GROUPS))

        valid_adv = b_advantages[b_valid > 0]
        b_advantages = (b_advantages - valid_adv.mean()) / (valid_adv.std() + 1e-8)

        num_samples = b_states.shape[0]
        # Running statistics stay on the device; one host read at the end
        # instead of a GPU sync per minibatch.
        # actor, critic, distill, entropy, clip fraction, approximate KL
        acc = torch.zeros(6, device=self.device)
        finite = torch.ones((), dtype=torch.bool, device=self.device)
        aux_acc = {}
        batches = 0
        distill_batch = min(self.cfg.batch_size, len(distill["states"])) if distill is not None else 0

        for _ in range(self.cfg.epochs):
            perm = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, self.cfg.batch_size):
                idx = perm[start:start + self.cfg.batch_size]
                extra = ()
                if distill is not None:
                    d_idx = torch.randint(len(distill["states"]), (distill_batch,), device=self.device)
                    extra = (distill["states"][d_idx], distill["masks"][d_idx], distill["targets"][d_idx])

                loss, actor_loss, critic_loss, distill_loss, stats, ok, pstats = self.losses(
                    b_states[idx], b_masks[idx], b_actions[idx], b_logprobs[idx], b_advantages[idx],
                    b_returns[idx], b_valid[idx], b_aux[idx], b_next[idx], b_turns[idx], b_out[idx],
                    b_amask[idx], *extra)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                for name, value in stats.items():
                    aux_acc[name] = aux_acc.get(name, 0.0) + value
                if distill_loss is not None:
                    acc[2] += distill_loss
                acc[0] += actor_loss
                acc[1] += critic_loss
                acc[3] += pstats["entropy"]
                acc[4] += pstats["clip_fraction"]
                acc[5] += pstats["approx_kl"]
                finite &= ok
                batches += 1

        if not bool(finite):
            raise RuntimeError("non-finite policy logits during optimisation")
        total_a_loss, total_c_loss, total_d_loss, ent, clipf, kl = acc.tolist()
        sums = {k: float(v) for k, v in aux_acc.items()}
        aux = {k: v / batches for k, v in sums.items() if not k.startswith("_")}
        aux["precision"] = sums.get("_hits", 0.0) / max(sums.get("_total", 0.0), 1.0)
        aux["baseline"] = sums.get("_baseline", 0.0) / max(sums.get("_total", 0.0), 1.0)
        policy = {"Entropy": ent / batches, "ClipFraction": clipf / batches, "ApproxKL": kl / batches}
        return total_a_loss / batches, total_c_loss / batches, total_d_loss / batches, aux, policy

    def evaluate(self, global_step):
        self.model.eval()
        agent = ModelPolicy(self.model, self.device)
        n = self.cfg.eval_games

        # Deck-only and the blind greedy match were dropped: the first is a
        # guaranteed win under round scoring and the second tracks the sighted
        # score within noise, so they cost a third of the eval for nothing.
        hs = self.cfg.hand_size
        vs_random = play_matches(agent, RandomPolicy(global_step), n, seed=global_step, hand_size=hs)
        vs_greedy = play_matches(agent, GreedyPolicy(global_step), n, seed=global_step + 3, hand_size=hs)
        greedy_score = vs_greedy["win_rate"] + 0.5 * vs_greedy["draw_rate"]
        self.writer.add_scalar("Eval/WinRate_vs_Random", vs_random["win_rate"], global_step)
        self.writer.add_scalar("Eval/Score_vs_Greedy", greedy_score, global_step)
        self.writer.add_scalar("Eval/PenaltyRate", vs_greedy["penalty_rate"], global_step)
        self.writer.add_scalar("Eval/DeepDrawRate", vs_greedy["deep_draw_rate"], global_step)
        self.writer.add_scalar("Eval/PileTakeRate", vs_greedy["pile_take_rate"], global_step)
        self.writer.add_scalar("Eval/MeldPoints", vs_greedy["meld_points"], global_step)
        self.writer.add_scalar("Eval/MeanTurns", vs_greedy["mean_turns"], global_step)
        line = (f"  Eval: vs random {vs_random['win_rate']:.1%} | "
                f"vs greedy {greedy_score:.1%} | pile-take {vs_greedy['pile_take_rate']:.1%} | "
                f"meld pts {vs_greedy['meld_points']:.1f} | {vs_greedy['mean_turns']:.1f} turns")

        # Win rate against a snapshot of this policy from frozen_refresh evals ago;
        # above 50% means self-play is still making progress.
        if self.frozen_model is not None:
            vs_frozen = play_matches(agent, ModelPolicy(self.frozen_model, self.device), n, seed=global_step + 2,
                                     hand_size=self.cfg.hand_size)
            score = vs_frozen["win_rate"] + 0.5 * vs_frozen["draw_rate"]
            self.writer.add_scalar("Eval/Score_vs_Frozen", score, global_step)
            line += f" | vs frozen {score:.1%}"
        if self.evals_run % self.cfg.frozen_refresh == 0:
            self.frozen_model = copy.deepcopy(self.model)
        line += self.update_best(agent, global_step, n)
        if self.cfg.search_eval_every and self.evals_run % self.cfg.search_eval_every == 0:
            line += self.evaluate_search(agent, global_step)
        self.evals_run += 1

        self.model.train()
        print(line)

    def update_best(self, agent, global_step, n):
        """Promote the live policy to best.pth when it beats the incumbent
        head-to-head by the configured margin. Head-to-head is a stronger
        signal than the greedy score once the policy is well past greedy."""
        first = self.best_model is None
        if first:
            promoted = True
            note = f" | best: first checkpoint -> {self.cfg.best_checkpoint}"
        else:
            vs_best = play_matches(agent, ModelPolicy(self.best_model, self.device), n, seed=global_step + 6,
                                   hand_size=self.cfg.hand_size)
            score = vs_best["win_rate"] + 0.5 * vs_best["draw_rate"]
            self.writer.add_scalar("Eval/Score_vs_Best", score, global_step)
            promoted = score >= 0.5 + self.cfg.best_margin
            note = f" | vs best {score:.1%} " + ("(promoted)" if promoted else f"(best is step {self.best_step:,})")
        if promoted:
            self.best_model = copy.deepcopy(self.model).eval()
            self.best_step = global_step
            self.save_checkpoint(self.cfg.best_checkpoint)
        self.writer.add_scalar("Eval/BestStep", self.best_step, global_step)
        return note

    def evaluate_search(self, plain, global_step):
        cfg = self.cfg
        search = SearchPolicy(self.model, self.device, worlds=cfg.search_worlds,
                              max_actions=cfg.search_actions, seed=global_step,
                              horizon=cfg.search_horizon, endgame=cfg.search_endgame,
                              reward_scale=cfg.reward_scale, replies=cfg.search_replies,
                              max_sims=cfg.search_max_sims)
        t0 = time.time()
        vs_plain = play_matches(search, plain, cfg.search_eval_games, seed=global_step + 4, hand_size=cfg.hand_size)
        vs_greedy = play_matches(search, GreedyPolicy(global_step), cfg.search_eval_games, seed=global_step + 5,
                                 hand_size=cfg.hand_size)
        seconds_per_game = (time.time() - t0) / (2 * cfg.search_eval_games)
        stats = search.stats()

        plain_score = vs_plain["win_rate"] + 0.5 * vs_plain["draw_rate"]
        greedy_score = vs_greedy["win_rate"] + 0.5 * vs_greedy["draw_rate"]
        self.writer.add_scalar("Search/Score_vs_Plain", plain_score, global_step)
        self.writer.add_scalar("Search/Score_vs_Greedy", greedy_score, global_step)
        self.writer.add_scalar("Search/Agreement", stats["agreement"], global_step)
        self.writer.add_scalar("Search/ValueGap", stats["value_gap"], global_step)
        self.writer.add_scalar("Search/SecondsPerGame", seconds_per_game, global_step)
        return (f"\n  Search({cfg.search_worlds}x{cfg.search_actions}): vs plain {plain_score:.1%} | "
                f"vs greedy {greedy_score:.1%} | agrees with model {stats['agreement']:.1%} | "
                f"value gap {stats['value_gap']:+.1f} | {seconds_per_game:.2f}s/game")

    def save_checkpoint(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.model.state_dict(), path)

if __name__ == "__main__":
    cfg = PPOConfig()
    trainer = PPOTrainer(cfg)
    trainer.train()
