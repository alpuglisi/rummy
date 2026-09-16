from dataclasses import dataclass

@dataclass
class PPOConfig:
    # Environment Setup
    num_envs: int = 64
    num_steps: int = 512       # Steps per environment before an update
    env_threads: int = 1       # C++ threads for stepping envs; >1 only pays off with hundreds of envs
    obs_dim: int = 159
    action_dim: int = 105
    
    # Training Parameters
    total_timesteps: int = 50_000_000
    learning_rate: float = 3e-4
    epochs: int = 4
    batch_size: int = 256
    
    # PPO Math
    reward_scale: float = 0.02  # Engine rewards are +-100 at game end; keep value targets O(1)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01     
    vf_coef: float = 0.5       
    
    # Evaluation (win rate vs a random opponent, logged to TensorBoard)
    eval_interval: int = 5     # Run every N PPO updates
    eval_games: int = 256
    frozen_refresh: int = 4    # Re-snapshot the frozen self-play opponent every N evals

    # System
    device: str = "cuda"
