#!/bin/bash
#SBATCH --job-name=dz_mkuntr
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:90g
#SBATCH --tmp=120G
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/mkuntr.%j.log
# Build a NO-TRAINING joint-space baseline checkpoint (random LoRA + random robot MLPs,
# no warm-start, unfreeze=0, 1 step). Used as the "policy that has seen no training"
# baseline in eval_joints_trained_vs_untrained.sbatch.
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
export FRANKA_ORCA_DATA_ROOT="$REPO/data/franka_orca_lerobot_joints"
export OUTPUT_DIR="$REPO/checkpoints/dreamzero_franka_orca_lora_joints_untrained"
export UNFREEZE_TOP_K=0
export NUM_GPUS=1
export MAX_STEPS=1
export SAVE_STEPS=1
export REPORT_TO=none
test -d "$FRANKA_ORCA_DATA_ROOT/meta" || { echo "ERROR: $FRANKA_ORCA_DATA_ROOT/meta missing"; exit 1; }
# Fresh build only: refuse to clobber an existing (possibly resuming) dir.
if [ -e "$OUTPUT_DIR/config.json" ] || ls -d "$OUTPUT_DIR"/checkpoint-* >/dev/null 2>&1; then
  echo "==> $OUTPUT_DIR already has a checkpoint; skipping build (delete it to rebuild)."; exit 0
fi
nvidia-smi -L || true
bash "$REPO/scripts/train/franka_orca_build_untrained.sh"
echo "==> build script exit: $?"
ls -la "$OUTPUT_DIR"
