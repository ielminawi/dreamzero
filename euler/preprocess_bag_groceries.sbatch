#!/bin/bash
#SBATCH --job-name=dz_prep
#SBATCH --array=0-9%2
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:24g
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/prep.%A_%a.log

# Shard of the bag_groceries h5 -> LeRobot conversion (with optional bg removal).
# Array of NUM_SHARDS tasks; each task converts its round-robin episode subset on 1 GPU.
# Run finalize_gear.sbatch ONCE after the whole array succeeds.
#
# Submit with:
#   BG=segformer sbatch euler/preprocess_bag_groceries.sbatch     # bg removal ON
#   BG=none      sbatch euler/preprocess_bag_groceries.sbatch     # baseline, no bg
set -uo pipefail
module load eth_proxy 2>/dev/null || true
cd /cluster/scratch/rjiang/dreamzero
export HF_HOME=/cluster/scratch/rjiang/hf_cache
export TOKENIZERS_PARALLELISM=false
export DS_ACCELERATOR=cpu   # force deepspeed accelerator to CPU so transformers' deepspeed
                            # integration doesn't JIT CUDA ops (needs CUDA_HOME) at segformer import;
                            # bg removal still uses GPU via torch directly.

RAW=${RAW:-/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries}
DATA_ROOT=${DATA_ROOT:-/cluster/scratch/rjiang/dreamzero/data/franka_orca_lerobot}
NUM_SHARDS=${NUM_SHARDS:-10}
BG=${BG:-mask2former}
TASK=${TASK:-"bag the groceries"}

echo "==> node $(hostname); shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}; bg=${BG}; out=${DATA_ROOT}"
nvidia-smi -L
.venv/bin/python scripts/data/convert_bag_groceries.py \
  --input-dir "$RAW" --output-dir "$DATA_ROOT" \
  --fps 50 --target-resolution 640x480 --task "$TASK" \
  --num-shards "$NUM_SHARDS" --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --bg-removal "$BG" ${BG_MODEL:+--bg-model "$BG_MODEL"}
echo "==> shard ${SLURM_ARRAY_TASK_ID} exit: $?"
