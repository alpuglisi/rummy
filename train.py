from config import PPOConfig
from trainer import PPOTrainer
import torch

def main():
    # Load hyperparameter configurations
    config = PPOConfig()
    
    # You can override config defaults here for local testing
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.total_timesteps = 50_000_000 
    
    print(f"Initializing Rummy RL Training on {config.device.upper()}...")
    
    # Initialize and start the PPO Trainer
    trainer = PPOTrainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
    finally:
        # Always save the model weights when stopping
        trainer.save_checkpoint("rummy_agent_checkpoint.pth")
        print("Checkpoint saved. Shutting down environments.")
        trainer.envs.close()

if __name__ == "__main__":
    main()
