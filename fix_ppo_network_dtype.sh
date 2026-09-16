#!/usr/bin/env bash
# Run from the project root. Fixes the FP16 masking dtype bug in
# models/ppo_network.py: huge_negative was missing dtype=logits.dtype,
# so it silently defaulted to float32 regardless of the model's actual dtype.
set -euo pipefail

FILE="models/ppo_network.py"

if [[ ! -f "$FILE" ]]; then
    echo "error: $FILE not found (run this script from the project root)" >&2
    exit 1
fi

OLD='huge_negative = torch.tensor(torch.finfo(logits.dtype).min, device=logits.device)'
NEW='huge_negative = torch.tensor(torch.finfo(logits.dtype).min, dtype=logits.dtype, device=logits.device)'

if grep -qF "$NEW" "$FILE"; then
    echo "already fixed: $FILE"
    exit 0
fi

if ! grep -qF "$OLD" "$FILE"; then
    echo "error: expected line not found in $FILE (file may have changed); no changes made" >&2
    exit 1
fi

python3 - "$FILE" "$OLD" "$NEW" <<'PYEOF'
import sys
path, old, new = sys.argv[1:4]
with open(path, "r") as f:
    content = f.read()
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
PYEOF

echo "patched: $FILE"
