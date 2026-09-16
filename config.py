from dataclasses import dataclass

@dataclass
class PPOConfig:
    # Environment Setup
    num_envs: int = 512
    num_steps: int = 128       # Steps per environment before an update (65,536 samples)
    env_threads: int = 0       # C++ threads for stepping envs; 0 = one per CPU core
    blank_known_prob: float = 0.5  # Fraction of games that hide the opponent-known-cards channel
    obs_dim: int = 318         # Must match OBS_SPACE_SIZE in rummy_env.h
    action_dim: int = 105
    
    # Network
    hidden_size: int = 512
    num_layers: int = 4        # Residual blocks in the shared trunk; heads are hidden_size // 2 wide
    residual: bool = True      # LayerNorm residual trunk; False gives the original plain MLP

    # Training Parameters
    total_timesteps: int = 100_000_000
    learning_rate: float = 3e-4
    epochs: int = 4
    batch_size: int = 2048
    
    # PPO Math
    reward_scale: float = 0.02  # Engine rewards are +-100 at game end; keep value targets O(1)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01     
    vf_coef: float = 0.5       
    
    # Evaluation (win rate vs a random opponent, logged to TensorBoard)
    eval_interval: int = 75    # Run every N PPO updates
    eval_games: int = 1000
    frozen_refresh: int = 4    # Re-snapshot the frozen self-play opponent every N evals

    # Search diagnostics (model + Monte Carlo search vs the model alone)
    search_eval_every: int = 2 # Run every N evals; costlier than the plain evals
    search_eval_games: int = 100
    search_worlds: int = 16
    search_actions: int = 4

    # Expert iteration: every distill_every updates, run the search on live
    # training positions (rollouts always play to the end of the game) and pull
    # the policy toward the search's action distribution.
    distill_every: int = 8
    distill_positions: int = 256
    distill_worlds: int = 64       # +-7 pts per candidate outcome; 16 worlds was +-15, noisier than the temperature
    distill_actions: int = 4
    distill_coef: float = 0.5        # Weight of the distillation loss next to the PPO loss; 0 disables
    distill_temperature: float = 15.0  # Points; softmax over candidate outcomes / temperature

    # System
    device: str = "cuda"
