#!/bin/bash

SESSION_NAME="rummy_train"

# Check if the session is already running so we don't duplicate it
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "Error: Session '$SESSION_NAME' is already running."
    echo "Attach to it using: tmux attach -t $SESSION_NAME"
    exit 1
fi

echo "Starting training pipeline in tmux session: $SESSION_NAME..."

# 1. Create a new detached tmux session and name the first window 'training'
tmux new-session -d -s $SESSION_NAME -n 'training'

# 2. Send the command to run the Python training script (C-m simulates hitting Enter)
tmux send-keys -t $SESSION_NAME:0 'python train.py' C-m

# 3. Create a second window named 'tensorboard'
tmux new-window -t $SESSION_NAME -n 'tensorboard'

# 4. Send the command to launch TensorBoard
tmux send-keys -t $SESSION_NAME:1 'tensorboard --logdir=runs/ --port=6006' C-m

# 5. Switch focus back to the first window so it's what you see when you attach
tmux select-window -t $SESSION_NAME:0

echo "✅ Success! PyTorch and TensorBoard are now running in the background."
echo "➡️  To watch the training live, run: tmux attach -t $SESSION_NAME"
echo "➡️  To leave the session without stopping it, press: Ctrl + B, then release and press D."
