#!/bin/bash
#SBATCH --job-name=dz_train
#SBATCH --partition=gpupr.120h
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=nvidia_a100_80gb_pcie:2
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/train.%j.log

# NOTE: the ls_polle share is capped at 2 GPUs, so this uses 2x A100-80GB (not 8).
# ZeRO-2 replicates the LoRA-frozen 14B base per GPU, so per-GPU memory == the original
# 8x80GB run -> it still fits; you just get fewer optimizer steps within the 100h wall.

# DreamZero franka_orca_bimanual LoRA training, 100h, 8x A100-80GB, DeepSpeed ZeRO-2.
# Wraps scripts/train/franka_orca_training.sh (the maintained arg list) with the Euler
# env. max_steps is set very high so the 100h wall-clock is the stopping criterion;
# SAVE_STEPS guarantees checkpoints land within the window (LoRA-only ckpts are small).
#
# Alt GPUs if A100-80 is busy (override at submit time):
#   sbatch -p cuda13pr.120h --gpus=nvidia_rtx_pro_6000:8 euler/train_franka_orca_100h.sbatch
set -uo pipefail
module load eth_proxy cuda/13.0.2 2>/dev/null || true
cd /cluster/scratch/rjiang/dreamzero

export CUDA_HOME=${CUDA_HOME:-/cluster/software/manual/cuda/13.0.2/x86_64}
export PATH="$PWD/.venv/bin:$PATH"            # torchrun + huggingface-cli from .venv
export HF_HOME=/cluster/scratch/rjiang/hf_cache
export HF_HUB_OFFLINE=1                        # all weights are local
export TOKENIZERS_PARALLELISM=false
export ATTENTION_BACKEND=torch
export OMP_NUM_THREADS=8

export NUM_GPUS=${NUM_GPUS:-2}
export REPORT_TO=${REPORT_TO:-none}
export FRANKA_ORCA_DATA_ROOT=${DATA_ROOT:-/cluster/scratch/rjiang/dreamzero/data/franka_orca_bag_groceries_lerobot}
export OUTPUT_DIR=${OUTPUT_DIR:-/cluster/scratch/rjiang/dreamzero/checkpoints/dreamzero_franka_orca_lora_euler}
export WAN_CKPT_DIR=/cluster/scratch/rjiang/dreamzero/checkpoints/Wan2.1-I2V-14B-480P
export TOKENIZER_DIR=/cluster/scratch/rjiang/dreamzero/checkpoints/umt5-xxl
export PRETRAINED_PATH=/cluster/scratch/rjiang/dreamzero/checkpoints/DreamZero-AgiBot
export MAX_STEPS=${MAX_STEPS:-1000000}         # wall-clock (100h) governs, not step count
export SAVE_STEPS=${SAVE_STEPS:-2000}          # frequent small LoRA checkpoints

echo "==> node $(hostname); data=$FRANKA_ORCA_DATA_ROOT; out=$OUTPUT_DIR"
nvidia-smi -L
test -d "$FRANKA_ORCA_DATA_ROOT/meta" || { echo "ERROR: dataset/meta missing -- run preprocessing first"; exit 1; }
bash scripts/train/franka_orca_training.sh
echo "==> train exit: $?"
