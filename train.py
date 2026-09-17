import argparse
import glob
import os
import shutil

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


def clean_logs(log_dir):
    """Remove the previous run's TensorBoard event files so its curves do not
    merge into the new run's. TensorBoard shows every event file in a run
    directory as one run, which mixes old and new data."""
    if not os.path.isdir(log_dir):
        return
    stale = [entry for entry in os.listdir(log_dir)]
    for name in stale:
        path = os.path.join(log_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    if stale:
        print(f"Removed {len(stale)} TensorBoard file(s) from the previous run in {log_dir}/.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="do not delete checkpoints left by a previous run")
    parser.add_argument("--keep-logs", action="store_true",
                        help="do not delete TensorBoard event files left by a previous run")
    parser.add_argument("--no-compile", action="store_true",
                        help="run the optimiser's minibatch losses eagerly instead of through torch.compile")
    parser.add_argument("--no-cuda-graphs", action="store_true",
                        help="run the opponent pool's forwards eagerly instead of replaying CUDA graphs")
    args = parser.parse_args()

    config = PPOConfig()
    if args.no_compile:
        config.compile_optimize = False
    if args.no_cuda_graphs:
        config.pool_cuda_graphs = False

    if not args.keep_checkpoints:
        clean_checkpoints()
    if not args.keep_logs:
        clean_logs(config.log_dir)

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
        if trainer.best_model is not None:
            print(f"Best checkpoint (from step {trainer.best_step:,}): {config.best_checkpoint}")
        trainer.writer.close()
        trainer.envs.close()

if __name__ == "__main__":
    main()
