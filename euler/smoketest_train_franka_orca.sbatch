#!/bin/bash
#SBATCH --job-name=dz_smoke_train
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --tmp=120G
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/smoke_train.%j.log
# GPU smoke test: 3 training steps on the 3-episode mini dataset. Validates the
# real GR00T dataloader + model build + pretrained load + deepspeed zero2 +
# forward/backward on the RTX PRO 6000 (Blackwell) before the 100h run. Exits
# non-zero on any failure -> the 100h job (afterok) auto-cancels.
set -eo pipefail
module load eth_proxy 2>/dev/null || true
REPO=/cluster/scratch/rjiang/dreamzero
cd "$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export ATTENTION_BACKEND=torch          # SDPA: Blackwell (sm_120) safe, no flash-attn kernels
export DS_ACCELERATOR=cuda
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
export HF_HOME=/cluster/scratch/rjiang/hf_cache
mkdir -p "$HF_HOME"
# Stage AgiBot (43GB) to node-local NVMe (Lustre safetensors mmap is ~8MB/s).
if [ -n "${TMPDIR:-}" ]; then
  echo "staging DreamZero-AgiBot -> $TMPDIR/dz_agibot ..."
  mkdir -p "$TMPDIR/dz_agibot"
  cp -r "$REPO/checkpoints/DreamZero-AgiBot/." "$TMPDIR/dz_agibot/"
  export PRETRAINED_PATH="$TMPDIR/dz_agibot"
fi
export FRANKA_ORCA_DATA_ROOT="$REPO/data/franka_orca_lerobot_mini"
export OUTPUT_DIR="$REPO/checkpoints/dz_smoke_${SLURM_JOB_ID}"
export NUM_GPUS=1
export MAX_STEPS=3
export SAVE_STEPS=1000000
export REPORT_TO=none
nvidia-smi -L || true
bash "$REPO/scripts/train/franka_orca_training.sh"
echo "==> SMOKE TRAIN PASSED"
