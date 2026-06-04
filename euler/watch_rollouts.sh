#!/bin/bash
# Submit a sim rollout for each NEW training checkpoint (dedup via a marker file),
# logging each rollout video to wandb. Run periodically while training, e.g.:
#   /loop 30m bash euler/watch_rollouts.sh
# or from a login shell every so often. Rollouts use the 2nd GPU (ls_polle cap=2).
set -uo pipefail
REPO=/cluster/scratch/rjiang/dreamzero
OUT="${OUTPUT_DIR:-$REPO/checkpoints/dreamzero_franka_orca_lora}"
[ -d "$OUT" ] || { echo "no checkpoints yet at $OUT"; exit 0; }
n=0
for ck in "$OUT"/checkpoint-*/; do
  ck="${ck%/}"
  [ -f "$ck/model.safetensors" ] || continue
  [ -e "$ck/.rolledout" ] && continue
  echo "submitting rollout for $ck"
  CKPT="$ck" sbatch "$REPO/euler/rollout_checkpoint.sbatch" && touch "$ck/.rolledout" && n=$((n+1))
done
echo "submitted $n new rollout(s)"
