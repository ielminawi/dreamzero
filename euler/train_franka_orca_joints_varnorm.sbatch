#!/bin/bash
#SBATCH --job-name=dz_jvarn
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:90g
#SBATCH --tmp=120G
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/train_jvarnorm.%j.log
# JOINT-SPACE retrain with the VARIANCE-NORMALIZED action loss (durable fix for the
# under-learned within-chunk MOTION). Warm-starts the EE r16/checkpoint-6000 VIDEO/DiT
# adaptation but trains the joint action head FROM STEP 0 with a FRESH head
# (warmstart_skip_action_head_mlps=true drops the inherited EE MLPs). Enables
# action_loss_variance_normalize so every action dim contributes equally and the small
# task-relevant motions finally get gradient. Auto-resumes from its own latest checkpoint.
set -eo pipefail
module load eth_proxy cuda/13.0.2 2>/dev/null || true
REPO=/cluster/scratch/rjiang/dreamzero
cd "$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export CUDA_HOME=${CUDA_HOME:-/cluster/software/manual/cuda/13.0.2/x86_64}
export ATTENTION_BACKEND=torch
export DS_ACCELERATOR=cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
export HF_HOME=/cluster/scratch/rjiang/hf_cache
mkdir -p "$HF_HOME"
if [ -n "${TMPDIR:-}" ]; then
  echo "staging DreamZero-AgiBot -> $TMPDIR/dz_agibot ..."
  mkdir -p "$TMPDIR/dz_agibot"
  cp -r "$REPO/checkpoints/DreamZero-AgiBot/." "$TMPDIR/dz_agibot/"
  export PRETRAINED_PATH="$TMPDIR/dz_agibot"
fi
export FRANKA_ORCA_DATA_ROOT="$REPO/data/franka_orca_lerobot_joints_clean"
export OUTPUT_DIR="$REPO/checkpoints/dreamzero_franka_orca_lora_joints_varnorm"
export WARMSTART_LORA_PATH="$REPO/checkpoints/dreamzero_franka_orca_lora_r16/checkpoint-6000"
export UNFREEZE_TOP_K=3
export NUM_GPUS=1
export MAX_STEPS=${MAX_STEPS:-14000}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
export REPORT_TO=wandb
export WANDB_PROJECT=dreamzero
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=/cluster/scratch/rjiang/dreamzero/wandb
export WANDB_RUN_NAME="franka_orca_joints_varnorm_$(date +%Y%m%d_%H%M 2>/dev/null || echo run)"
mkdir -p "$WANDB_DIR"
test -f "$FRANKA_ORCA_DATA_ROOT/meta/relative_stats_dreamzero.json" || { echo "ERROR: cleaned dataset stats missing"; exit 1; }
test -d "$WARMSTART_LORA_PATH" || { echo "ERROR: warm-start ckpt missing: $WARMSTART_LORA_PATH"; exit 1; }
nvidia-smi -L || true
bash "$REPO/scripts/train/franka_orca_training_joints_varnorm.sh"
echo "==> training script exit: $?"
