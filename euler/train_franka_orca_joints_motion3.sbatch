#!/bin/bash
#SBATCH --job-name=dz_jmot3
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:90g
#SBATCH --tmp=120G
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/train_jmot3.%j.log
# JOINT-SPACE motion-loss continuation (motion2 recipe, v3 leg). The motion2 run's LR schedule
# fully decayed to ~0 at step 1500 while train loss was STILL descending (0.20 -> 0.15), so this
# is NOT a resume (schedule is dead) but a FRESH-schedule WARM-START from motion2/checkpoint-1500
# (keeps the trained joint head + hand-push weighting). Same motion2 recipe/loss; new warmup+decay
# over MAX_STEPS more steps to push the still-descending loss further. Separate OUTPUT_DIR so the
# motion2 1500-step lineage is preserved untouched.
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
export OUTPUT_DIR="$REPO/checkpoints/dreamzero_franka_orca_lora_joints_motion3"
export WARMSTART_LORA_PATH="$REPO/checkpoints/dreamzero_franka_orca_lora_joints_motion2/checkpoint-1500"
export UNFREEZE_TOP_K=3
export NUM_GPUS=1
export MOTION_W=${MOTION_W:-2.0}
export MAX_STEPS=${MAX_STEPS:-1500}
export SAVE_STEPS=${SAVE_STEPS:-500}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}   # code asserts >=5; run only saves 3 (steps 500/1000/1500) so no extra disk
export REPORT_TO=wandb
export WANDB_PROJECT=dreamzero
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=/cluster/scratch/rjiang/dreamzero/wandb
export WANDB_RUN_NAME="franka_orca_joints_motion3_$(date +%Y%m%d_%H%M 2>/dev/null || echo run)"
mkdir -p "$WANDB_DIR"
test -f "$FRANKA_ORCA_DATA_ROOT/meta/relative_stats_dreamzero.json" || { echo "ERROR: cleaned dataset stats missing"; exit 1; }
test -d "$WARMSTART_LORA_PATH" || { echo "ERROR: warm-start ckpt missing: $WARMSTART_LORA_PATH"; exit 1; }
test -f "$WARMSTART_LORA_PATH/model.safetensors" || { echo "ERROR: warm-start model.safetensors missing"; exit 1; }
nvidia-smi -L || true
bash "$REPO/scripts/train/franka_orca_training_joints_motion2.sh"
echo "==> training script exit: $?"
