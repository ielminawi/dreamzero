#!/bin/bash
#SBATCH --job-name=dz_train_franka_orca_100h
#SBATCH --partition=cuda13pr.120h
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --tmp=120G
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/train_franka_orca.%j.log
# 100h LoRA finetune of DreamZero (Wan2.1-I2V-14B base) on bag_groceries
# (franka_orca_bimanual), 1x RTX PRO 6000 (96GB), deepspeed zero2. Checkpoints every
# 2000 steps (save_total_limit=10, LoRA-only). max_steps high -> 100h wall governs.
set -eo pipefail
module load eth_proxy cuda/13.0.2 2>/dev/null || true
REPO=/cluster/scratch/rjiang/dreamzero
cd "$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export CUDA_HOME=${CUDA_HOME:-/cluster/software/manual/cuda/13.0.2/x86_64}  # deepspeed JIT ops
export ATTENTION_BACKEND=torch          # SDPA: Blackwell (sm_120) safe
export DS_ACCELERATOR=cuda
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
export HF_HOME=/cluster/scratch/rjiang/hf_cache
mkdir -p "$HF_HOME"
if [ -n "${TMPDIR:-}" ]; then
  echo "staging DreamZero-AgiBot -> $TMPDIR/dz_agibot ..."
  mkdir -p "$TMPDIR/dz_agibot"
  cp -r "$REPO/checkpoints/DreamZero-AgiBot/." "$TMPDIR/dz_agibot/"
  export PRETRAINED_PATH="$TMPDIR/dz_agibot"
fi
export FRANKA_ORCA_DATA_ROOT="$REPO/data/franka_orca_lerobot"
export OUTPUT_DIR="$REPO/checkpoints/dreamzero_franka_orca_lora"
export NUM_GPUS=1
export MAX_STEPS=${MAX_STEPS:-1000000}   # 100h wall-clock governs, not step count
export SAVE_STEPS=2000
export REPORT_TO=none
test -d "$FRANKA_ORCA_DATA_ROOT/meta" || { echo "ERROR: $FRANKA_ORCA_DATA_ROOT/meta missing -- preprocessing must finish first"; exit 1; }
nvidia-smi -L || true
bash "$REPO/scripts/train/franka_orca_training.sh"
echo "==> training script exit: $?"
