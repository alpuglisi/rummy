import argparse
import glob
import os
import shutil

from config import PPOConfig
from trainer import PPOTrainer
import torch

CHECKPOINT_DIR = "checkpoints"
FINAL_CHECKPOINT = "rummy_agent_checkpoint.pth"


def clean_checkpoints(keep=None):
    """Remove the previous run's checkpoints, except `keep` (e.g. an
    --init-checkpoint the new run is about to warm-start from, which may
    itself live in checkpoints/)."""
    keep_abs = os.path.abspath(keep) if keep else None
    stale = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))
    if os.path.exists(FINAL_CHECKPOINT):
        stale.append(FINAL_CHECKPOINT)
    removed = 0
    for path in stale:
        if keep_abs and os.path.abspath(path) == keep_abs:
            continue
        os.remove(path)
        removed += 1
    if removed:
        print(f"Removed {removed} checkpoint(s) from the previous run.")


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
    parser.add_argument("--init-checkpoint", type=str, default="",
                        help="warm-start the live policy from this checkpoint (e.g. a previous run's best.pth) "
                             "instead of random weights; the optimizer, opponent pool and learning-rate schedule "
                             "still start fresh")
    args = parser.parse_args()

    if args.init_checkpoint and not os.path.isfile(args.init_checkpoint):
        parser.error(f"--init-checkpoint {args.init_checkpoint!r} does not exist")

    config = PPOConfig()
    if args.no_compile:
        config.compile_optimize = False
    if args.no_cuda_graphs:
        config.pool_cuda_graphs = False
    config.init_checkpoint = args.init_checkpoint

    if not args.keep_checkpoints:
        clean_checkpoints(keep=args.init_checkpoint or None)
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
