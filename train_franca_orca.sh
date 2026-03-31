#!/bin/bash
#SBATCH --job-name=dz-franka-orca
#SBATCH --output=logs/dz-franka-orca-%j.out
#SBATCH --error=logs/dz-franka-orca-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-task=2
#SBATCH --account=ls_polle
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rjiang@ethz.ch

# ============================================================
# Environment
# ============================================================
module load stack/2024-06 gcc/12.2.0
module load eth_proxy
source $SCRATCH/dreamzero/venv/bin/activate
cd $SCRATCH/dreamzero/dreamzero

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_HOME=/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.9.1-k2xznhlnmfrxk5ahn5h6axehok4bs7b2

mkdir -p logs

# ============================================================
# Launch training
# ============================================================
FRANKA_ORCA_DATA_ROOT=/cluster/scratch/rjiang/dreamzero/data/franka_orca_lerobot \
OUTPUT_DIR=$SCRATCH/dreamzero/output/dreamzero_franka_orca_lora \
WAN_CKPT_DIR=$SCRATCH/dreamzero/checkpoints/Wan2.1-I2V-14B-480P \
TOKENIZER_DIR=$SCRATCH/dreamzero/checkpoints/umt5-xxl \
    bash scripts/train/franka_orca_training.sh 