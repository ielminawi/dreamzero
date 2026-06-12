#!/bin/bash
#SBATCH --job-name=dz_smoke_unf4
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:90g
#SBATCH --tmp=120G
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/smoke_unf4.%j.log
# SMOKE TEST for the warm-start + unfreeze-top-4 run. Validates BEFORE the multi-day job:
#   (1) edits don't crash; (2) [warmstart] lands keys (LoRA delta loaded post-injection);
#   (3) [unfreeze] reports ~1.4B extra trainable params on top-4 blocks;
#   (4) 2 optimizer steps run without OOM on 80GB; (5) a checkpoint saves (file > LoRA-only size).
set -eo pipefail
module load eth_proxy cuda/13.0.2 2>/dev/null || true
REPO=/cluster/scratch/rjiang/dreamzero
cd "$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export CUDA_HOME=${CUDA_HOME:-/cluster/software/manual/cuda/13.0.2/x86_64}
export ATTENTION_BACKEND=torch
export DS_ACCELERATOR=cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reclaim fragmentation; top-4 unfreeze is ~300MB over 80GB without it
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
export OUTPUT_DIR="$REPO/checkpoints/_smoke_r16unf4"
export WARMSTART_LORA_PATH="$REPO/checkpoints/dreamzero_franka_orca_lora_r16/checkpoint-6000"
export UNFREEZE_TOP_K=3
export NUM_GPUS=1
export MAX_STEPS=2
export SAVE_STEPS=2
export REPORT_TO=none
export WANDB_MODE=offline
rm -rf "$OUTPUT_DIR"   # always a fresh throwaway dir
nvidia-smi -L || true
bash "$REPO/scripts/train/franka_orca_training_unf4.sh"
echo "==> SMOKE exit: $?"
echo "=== saved checkpoint size (should be >> 332MB LoRA-only if unfrozen blocks persisted) ==="
ls -laR "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | grep -E "model.safetensors|checkpoint-" | head
