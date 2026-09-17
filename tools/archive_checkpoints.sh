#!/usr/bin/env bash
# Copy the N most recently modified .pth files from checkpoints/ into archive/,
# so the next training run's opponent pool (config.pool_init_dir) starts
# seeded with recent, strong checkpoints instead of empty.
#
# Usage: tools/archive_checkpoints.sh [count] [checkpoint_dir] [archive_dir]
#   count           number of files to copy (default 10)
#   checkpoint_dir  where checkpoints are read from (default checkpoints)
#   archive_dir     where they're copied to (default archive)
set -euo pipefail

COUNT="${1:-10}"
CHECKPOINT_DIR="${2:-checkpoints}"
ARCHIVE_DIR="${3:-archive}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "error: checkpoint directory '$CHECKPOINT_DIR' does not exist" >&2
    exit 1
fi

mkdir -p "$ARCHIVE_DIR"

mapfile -t files < <(find "$CHECKPOINT_DIR" -maxdepth 1 -name "*.pth" -printf "%T@ %p\n" \
    | sort -rn | head -n "$COUNT" | cut -d' ' -f2-)

if [ "${#files[@]}" -eq 0 ]; then
    echo "error: no .pth files found in $CHECKPOINT_DIR/" >&2
    exit 1
fi

for f in "${files[@]}"; do
    cp -v "$f" "$ARCHIVE_DIR/"
done

echo "Copied ${#files[@]} checkpoint(s) from $CHECKPOINT_DIR/ to $ARCHIVE_DIR/"
