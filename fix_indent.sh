#!/bin/bash

# 1. Delete everything from the broken 'def save_checkpoint' line to the end of the file
sed -i '/def save_checkpoint/,$d' trainer.py

# 2. Append the perfectly indented code block back to the file
cat << 'PYTHON' >> trainer.py

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)

if __name__ == "__main__":
    import os
    os.makedirs("checkpoints", exist_ok=True)
    
    cfg = PPOConfig()
    trainer = PPOTrainer(cfg)
    trainer.train()
PYTHON

echo "✅ trainer.py repaired!"
