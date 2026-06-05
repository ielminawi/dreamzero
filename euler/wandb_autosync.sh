#!/bin/bash
# Keep the offline wandb run(s) synced to wandb.ai every 10 min for a near-live dashboard.
# Run detached on a LOGIN node (has internet):
#   setsid bash /cluster/scratch/rjiang/dreamzero/euler/wandb_autosync.sh >/cluster/scratch/rjiang/dreamzero/euler/logs/wandb_autosync.log 2>&1 < /dev/null &
# Stop it with:  pkill -f wandb_autosync
cd /cluster/scratch/rjiang/dreamzero || exit 1
export WANDB_ENTITY=${WANDB_ENTITY:-idsc-rda}
while true; do
  for d in checkpoints/dreamzero_franka_orca_lora/wandb/offline-run-* \
           checkpoints/dreamzero_franka_orca_lora_smoke/wandb/offline-run-*; do
    [ -d "$d" ] && .venv/bin/wandb sync --include-offline "$d" 2>&1 | grep -iE "done|error" | tail -1
  done
  echo "[autosync $(date '+%H:%M:%S')] synced; sleeping 600s"
  sleep 600
done
