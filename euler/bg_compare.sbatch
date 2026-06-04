#!/bin/bash
#SBATCH --job-name=dz_bgcmp
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:24g
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/bgcmp.%j.log

set -uo pipefail
module load eth_proxy 2>/dev/null || true
cd /cluster/scratch/rjiang/dreamzero
export HF_HOME=/cluster/scratch/rjiang/hf_cache
export TOKENIZERS_PARALLELISM=false
export DS_ACCELERATOR=cpu
echo "==> node $(hostname)"; nvidia-smi -L
.venv/bin/python scripts/data/bg_compare.py
echo "==> bg_compare exit: $?"
