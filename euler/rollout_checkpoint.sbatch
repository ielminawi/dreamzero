#!/bin/bash
#SBATCH --job-name=dz_rollout
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/rollout.%j.log

# Roll out ONE training checkpoint in the Isaac sim closed-loop eval and log the
# rollout video to wandb. Uses the 2nd GPU (ls_polle cap=2), so it runs alongside
# the 1-GPU training.
#   CKPT=/cluster/scratch/rjiang/dreamzero/checkpoints/dreamzero_franka_orca_lora/checkpoint-2000 \
#     sbatch euler/rollout_checkpoint.sbatch
set -uo pipefail
module load eth_proxy 2>/dev/null || true
cd /cluster/scratch/rjiang/dreamzero
: "${CKPT:?set CKPT=/path/to/checkpoint-XXXX}"
export DZ_LORA="$CKPT"
export DZ_EPISODES="${DZ_EPISODES:-2}"
export DZ_HORIZON="${DZ_HORIZON:-24}"
export DZ_INSTRUCTION="${DZ_INSTRUCTION:-bag the groceries}"
export WANDB_PROJECT="${WANDB_PROJECT:-dreamzero}"
export WANDB_ENTITY="${WANDB_ENTITY:-idsc-rda}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR=/cluster/scratch/rjiang/dreamzero/wandb
mkdir -p "$WANDB_DIR"

echo "==> rollout checkpoint: $CKPT"
[ -f "$CKPT/model.safetensors" ] || { echo "ERROR: $CKPT/model.safetensors missing"; exit 1; }
bash euler/run_e2e.sh; rc=$?
echo "==> run_e2e exit: $rc"
.venv/bin/python scripts/log_rollout_wandb.py --ckpt "$CKPT" || true
echo "==> rollout done"
