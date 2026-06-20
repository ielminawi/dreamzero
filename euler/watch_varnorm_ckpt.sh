#!/bin/bash
# Polls for the first complete varnorm checkpoint and auto-submits the decisive eval.
set -u
REPO=/cluster/scratch/rjiang/dreamzero
CKPT_DIR=$REPO/checkpoints/dreamzero_franka_orca_lora_joints_varnorm
DATA=$REPO/data/franka_orca_lerobot_joints_clean
TARGET=checkpoint-1000
while true; do
  C="$CKPT_DIR/$TARGET"
  if [ -f "$C/model.safetensors" ]; then
    # wait for size to stabilize (write finished)
    s1=$(stat -c%s "$C/model.safetensors" 2>/dev/null); sleep 30
    s2=$(stat -c%s "$C/model.safetensors" 2>/dev/null)
    if [ "$s1" = "$s2" ] && [ "${s1:-0}" -gt 1000000000 ]; then
      echo "[watch] $TARGET complete ($s2 bytes). Submitting eval $(date)."
      CKPT="$C" DATA="$DATA" OUT="$REPO/output/action_eval_varnorm_1000" \
        sbatch "$REPO/euler/eval_robuststats.sbatch"
      echo "[watch] eval submitted; exiting watcher."
      exit 0
    fi
  fi
  sleep 120
done
