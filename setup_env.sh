#!/usr/bin/env bash
# One-shot environment setup for a fresh EC2 instance.
# Installs system build tools, Python dependencies, builds the rummy_engine
# C++ extension, and verifies everything imports. Safe to re-run.
# Usage: ./setup_env.sh          (run from the project root)
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f setup.py || ! -f rummy_env.cpp ]]; then
    echo "error: run this from the rummy project root" >&2
    exit 1
fi

SUDO=""
if [[ $EUID -ne 0 ]] && command -v sudo >/dev/null; then
    SUDO="sudo"
fi

echo "==> Installing system packages"
if command -v dnf >/dev/null; then
    $SUDO dnf install -y -q gcc gcc-c++ python3-devel tmux git
elif command -v yum >/dev/null; then
    $SUDO yum install -y -q gcc gcc-c++ python3-devel tmux git
elif command -v apt-get >/dev/null; then
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq build-essential python3-dev tmux git
else
    echo "warning: unknown package manager; make sure g++ and Python headers are installed" >&2
fi

PY="${PYTHON:-python3}"
echo "==> Using $($PY --version) at $(command -v "$PY")"

echo "==> Installing Python packages"
$PY -m pip install -q setuptools wheel numpy pybind11 tensorboard pygame

if $PY -c "import torch" 2>/dev/null; then
    echo "    torch already installed: $($PY -c 'import torch; print(torch.__version__)')"
else
    # The default PyPI wheel bundles CUDA; use the CPU index when there is no GPU.
    if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
        $PY -m pip install -q torch
    else
        echo "    no GPU detected, installing CPU-only torch"
        $PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    fi
fi

echo "==> Building rummy_engine C++ extension"
rm -rf build  # setuptools skips the rebuild after header-only changes otherwise
$PY -m pip install -q --no-build-isolation --force-reinstall --no-deps .

echo "==> Verifying"
$PY - <<'EOF'
import numpy, torch, rummy_engine
print(f"    torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    gpu: {torch.cuda.get_device_name(0)}")
env = rummy_engine.RummyEnv(0)
state, mask = env.get_state(), env.get_legal_actions()
from config import PPOConfig
assert state.shape == (PPOConfig.obs_dim,) and mask.shape == (PPOConfig.action_dim,), state.shape
for _ in range(20):
    legal = numpy.flatnonzero(env.get_legal_actions())
    _, done = env.step(int(legal[0]))
    if done:
        env.reset()
print("    rummy_engine OK")
from trainer import PPOTrainer  # noqa: F401  (checks all project imports resolve)
print("    project imports OK")
EOF

echo
echo "Setup complete. Start training with: ./start_training.sh"
