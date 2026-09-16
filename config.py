from dataclasses import dataclass

@dataclass
class PPOConfig:
    # Environment Setup
    num_envs: int = 512
    num_steps: int = 128       # Steps per environment before an update (65,536 samples)
    env_threads: int = 0       # C++ threads for stepping envs; 0 = one per CPU core
    blank_known_prob: float = 0.5  # Fraction of games that hide the opponent-known-cards channel
    obs_dim: int = 266         # Must match OBS_SPACE_SIZE in rummy_env.h
    action_dim: int = 105
    
    # Network
    hidden_size: int = 512
    num_layers: int = 3        # Shared trunk depth; heads are hidden_size // 2 wide

    # Training Parameters
    total_timesteps: int = 50_000_000
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
    eval_interval: int = 25    # Run every N PPO updates
    eval_games: int = 1000
    frozen_refresh: int = 4    # Re-snapshot the frozen self-play opponent every N evals

    # System
    device: str = "cuda"
