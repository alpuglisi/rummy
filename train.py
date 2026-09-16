import argparse
import glob
import os

from config import PPOConfig
from trainer import PPOTrainer
import torch

CHECKPOINT_DIR = "checkpoints"
FINAL_CHECKPOINT = "rummy_agent_checkpoint.pth"


def clean_checkpoints():
    stale = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))
    if os.path.exists(FINAL_CHECKPOINT):
        stale.append(FINAL_CHECKPOINT)
    for path in stale:
        os.remove(path)
    if stale:
        print(f"Removed {len(stale)} checkpoint(s) from the previous run.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="do not delete checkpoints left by a previous run")
    args = parser.parse_args()

    if not args.keep_checkpoints:
        clean_checkpoints()

    config = PPOConfig()

    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Initializing Rummy RL Training on {config.device.upper()}...")

    trainer = PPOTrainer(config)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
    finally:
        trainer.save_checkpoint(FINAL_CHECKPOINT)
        print("Checkpoint saved. Shutting down environments.")
        trainer.writer.close()
        trainer.envs.close()

if __name__ == "__main__":
    main()
