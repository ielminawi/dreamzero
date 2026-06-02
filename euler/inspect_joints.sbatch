#!/bin/bash
#SBATCH --job-name=dz_inspect_joints
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:24g
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/inspect_joints.%j.log

# Validate the FrankaOrcaBimanual sim joint setup inside the Isaac container.
# No policy server needed — this only builds the env and inspects it.
set -uo pipefail
module load eth_proxy 2>/dev/null || true
source /cluster/scratch/rjiang/dreamzero/euler/scripts/_env.sh

mkdir -p "$DZ_OVHOME" "$DZ_LOGDIR"
if [ ! -f "$DZ_OVERLAY" ]; then
  echo "Creating Isaac writable overlay ($DZ_OVERLAY, 8GB)..."
  apptainer overlay create --size 8192 "$DZ_OVERLAY" 2>&1 || echo "WARN: overlay create failed"
fi

proxy="${DZ_PROXY:-http://proxy.service.consul:3128}"
args=( run --nv --env TERM=xterm
       --env "http_proxy=$proxy" --env "https_proxy=$proxy"
       --env "HTTP_PROXY=$proxy" --env "HTTPS_PROXY=$proxy"
       --env "no_proxy=localhost,127.0.0.1" --env "NO_PROXY=localhost,127.0.0.1"
       --home "$DZ_OVHOME:$DZ_OVHOME"
       --bind "$DZ_REPO:/app" --bind "$DZ_OUTPUT:/output" --pwd /app )
[ -f "$DZ_OVERLAY" ] && args+=( --overlay "$DZ_OVERLAY" )
args+=( "$DZ_SIF" eval_utils/inspect_joints.py --headless --enable_cameras
        --device "${DZ_SIM_DEVICE:-cpu}" )

echo "==> apptainer ${args[*]}"
apptainer "${args[@]}"
echo "==> inspect_joints exit code: $?"
