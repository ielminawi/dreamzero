#!/bin/bash
#SBATCH --job-name=dz_e2e
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/e2e.%j.log

# Full closed-loop e2e (policy server + Isaac Sim) on a 96GB GPU node, after the
# franka_orca joint-setup cfg fixes. Produces a rollout video.
set -uo pipefail
module load eth_proxy 2>/dev/null || true
export DZ_EPISODES="${DZ_EPISODES:-1}"
export DZ_HORIZON="${DZ_HORIZON:-24}"
export DZ_INSTRUCTION="${DZ_INSTRUCTION:-pick up the cube}"
bash /cluster/scratch/rjiang/dreamzero/euler/run_e2e.sh
echo "==> run_e2e exit code: $?"
