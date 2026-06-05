#!/bin/bash
#SBATCH --job-name=dz_smoke_train
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --tmp=120G
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/smoke_train.%j.log
# Short SMOKE test of the full training stack (model load + dataset loader on the new
# videos + fwd/bwd + DeepSpeed step + checkpoint save + wandb) on 1 RTX PRO 6000.
# 2h walltime backfills sooner than the 100h job; MAX_STEPS=10 / SAVE_STEPS=5 so it
# saves a checkpoint and exits quickly. Same script/env as the real run, so a PASS
# here means the 100h job will start cleanly.
set -eo pipefail
module load eth_proxy cuda/13.0.2 2>/dev/null || true
REPO=/cluster/scratch/rjiang/dreamzero
cd "$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export CUDA_HOME=${CUDA_HOME:-/cluster/software/manual/cuda/13.0.2/x86_64}
export ATTENTION_BACKEND=torch
export DS_ACCELERATOR=cuda
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
export HF_HOME=/cluster/scratch/rjiang/dreamzero/hf_cache
mkdir -p "$HF_HOME"
if [ -n "${TMPDIR:-}" ]; then
  mkdir -p "$TMPDIR/dz_agibot"; cp -r "$REPO/checkpoints/DreamZero-AgiBot/." "$TMPDIR/dz_agibot/"
  export PRETRAINED_PATH="$TMPDIR/dz_agibot"
fi
export FRANKA_ORCA_DATA_ROOT="$REPO/data/franka_orca_lerobot"
export OUTPUT_DIR="$REPO/checkpoints/dreamzero_franka_orca_lora_smoke"
export NUM_GPUS=1
export MAX_STEPS=10
export SAVE_STEPS=5
export REPORT_TO=wandb
export WANDB_PROJECT=dreamzero
export WANDB_ENTITY=${WANDB_ENTITY:-idsc-rda}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=/cluster/scratch/rjiang/dreamzero/wandb
export WANDB_RUN_NAME=smoke_$SLURM_JOB_ID
mkdir -p "$WANDB_DIR"
test -d "$FRANKA_ORCA_DATA_ROOT/meta" || { echo "ERROR: dataset missing"; exit 1; }
nvidia-smi -L
echo "==> SMOKE: max_steps=$MAX_STEPS save_steps=$SAVE_STEPS out=$OUTPUT_DIR"
bash "$REPO/scripts/train/franka_orca_training.sh"
echo "==> smoke exit: $?"
echo "==> checkpoints written:"; ls -1 "$OUTPUT_DIR" 2>/dev/null || echo "(none!)"
