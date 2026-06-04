#!/bin/bash
#SBATCH --job-name=dz_bggallery
#SBATCH --time=00:25:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:20g
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/bggallery.%j.log

# Background-removal before/after gallery (for human review before full preprocessing).
# Runs the .venv python directly on a GPU node (no Isaac container, no cuda module:
# torch uses its bundled cu129 runtime; loading cuda/13 would clash).
set -uo pipefail
module load eth_proxy 2>/dev/null || true
cd /cluster/scratch/rjiang/dreamzero
export HF_HOME=/cluster/scratch/rjiang/hf_cache
export TOKENIZERS_PARALLELISM=false
export DS_ACCELERATOR=cpu   # deepspeed is installed (for training); force its accelerator to CPU so
                            # transformers' deepspeed integration doesn't try to JIT CUDA ops (needs
                            # CUDA_HOME) at import. bg removal still runs on GPU via torch directly.
echo "==> node $(hostname); gpu:"; nvidia-smi -L
.venv/bin/python scripts/data/bg_gallery.py
echo "==> bg_gallery exit: $?"
