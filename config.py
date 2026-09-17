from dataclasses import dataclass

@dataclass
class PPOConfig:
    # Environment Setup
    num_envs: int = 1024       # Per-step cost is mostly fixed overhead, so more envs = more samples/s
    num_steps: int = 128       # Steps per environment before an update (131,072 samples)
    env_threads: int = 0       # C++ threads for stepping envs; 0 = one per CPU core
    blank_known_prob: float = 0.5  # Fraction of games that hide the opponent-known-cards channel
    obs_dim: int = 320         # Must match OBS_SPACE_SIZE in rummy_env.h
    action_dim: int = 105
    
    # Network
    hidden_size: int = 512
    num_layers: int = 4        # Residual blocks in the shared trunk; heads are hidden_size // 2 wide
    residual: bool = True      # LayerNorm residual trunk; False gives the original plain MLP

    # Training Parameters
    total_timesteps: int = 100_000_000
    learning_rate: float = 3e-4
    lr_final: float = 1e-5           # Linear anneal target; reached at the end of the run
    lr_anneal_start: float = 0.5     # Fraction of the run after which annealing begins

    # Opponent pool: a fraction of games are played against frozen earlier
    # policies instead of the live policy, so self-play cannot overfit to
    # itself. The learner only trains on its own moves in those games.
    pool_fraction: float = 0.5       # Fraction of games with a pool opponent; 0 = pure self-play
    pool_size: int = 8               # Snapshots kept (oldest dropped)
    pool_add_every: int = 50         # Updates between snapshots of the live policy (~6.5M steps)
    pool_init_dir: str = "archive"   # Checkpoints loaded into the pool at start, if the directory exists
    epochs: int = 4
    batch_size: int = 4096     # 32 minibatches per epoch at 1024 envs x 128 steps
    
    # PPO Math
    reward_scale: float = 0.02  # Engine rewards are +-100 at game end; keep value targets O(1)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01     
    vf_coef: float = 0.5       
    
    # Evaluation (win rate vs a random opponent, logged to TensorBoard)
    eval_interval: int = 75    # Run every N PPO updates (~10M steps at 1024 envs)
    eval_games: int = 1000
    frozen_refresh: int = 4    # Re-snapshot the frozen self-play opponent every N evals

    # Search diagnostics (model + Monte Carlo search vs the model alone)
    search_eval_every: int = 2 # Run every N evals; costlier than the plain evals
    search_eval_games: int = 100
    # Best-checkpoint tracking: at every eval the live policy plays eval_games
    # head-to-head against the incumbent best; it replaces it (and is written
    # to best_checkpoint) when it scores at least 50% + best_margin.
    best_checkpoint: str = "checkpoints/best.pth"
    best_margin: float = 0.02
    search_worlds: int = 16
    search_actions: int = 4

    # Expert iteration: every distill_every updates, run the search on live
    # training positions (rollouts always play to the end of the game) and pull
    # the policy toward the search's action distribution.
    distill_every: int = 4
    distill_positions: int = 256
    distill_worlds: int = 64       # ~+-7 pts per candidate outcome against a 15-pt temperature; 128 was half the run's wall clock
    distill_actions: int = 6
    distill_coef: float = 0.5        # Weight of the distillation loss next to the PPO loss; 0 disables
    distill_belief: bool = True      # Deal search worlds from the opponent-hand head once it is useful...
    distill_belief_min_gain: float = 0.10  # ...i.e. once Aux/TopKPrecision exceeds the random baseline by this much
    aux_coef: float = 0.5            # Weight of the opponent-hand prediction loss; 0 disables
    distill_temperature: float = 15.0  # Points; softmax over candidate outcomes / temperature

    # System
    device: str = "cuda"
    log_dir: str = "runs/rummy_ppo"   # TensorBoard event files; train.py clears it at the start of a run
