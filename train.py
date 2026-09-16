from config import PPOConfig
from trainer import PPOTrainer
import torch

def main():
    config = PPOConfig()
    
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.total_timesteps = 50_000_000 
    
    print(f"Initializing Rummy RL Training on {config.device.upper()}...")
    
    trainer = PPOTrainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
    finally:
        trainer.save_checkpoint("rummy_agent_checkpoint.pth")
        print("Checkpoint saved. Shutting down environments.")
        trainer.envs.close()

if __name__ == "__main__":
    main()
