#!/bin/bash
#SBATCH --job-name=dz_replay
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:24g
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/replay.%j.log

# Open-loop replay of recorded h5 actions in the Isaac sim (no policy server needed).
set -uo pipefail
module load eth_proxy 2>/dev/null || true
source /cluster/scratch/rjiang/dreamzero/euler/scripts/_env.sh
mkdir -p "$DZ_OVHOME" "$DZ_LOGDIR"
[ -f "$DZ_OVERLAY" ] || apptainer overlay create --size 8192 "$DZ_OVERLAY" 2>&1 || true
proxy="${DZ_PROXY:-http://proxy.service.consul:3128}"
args=( run --nv --env TERM=xterm
       --env "http_proxy=$proxy" --env "https_proxy=$proxy"
       --env "HTTP_PROXY=$proxy" --env "HTTPS_PROXY=$proxy"
       --env "no_proxy=localhost,127.0.0.1" --env "NO_PROXY=localhost,127.0.0.1"
       --home "$DZ_OVHOME:$DZ_OVHOME"
       --bind "$DZ_REPO:/app" --bind "$DZ_OUTPUT:/output"
       --bind /cluster/work/cvg/data/Egoverse:/cluster/work/cvg/data/Egoverse:ro
       --pwd /app )
[ -f "$DZ_OVERLAY" ] && args+=( --overlay "$DZ_OVERLAY" )
args+=( "$DZ_SIF" eval_utils/replay_h5.py --headless --enable_cameras
        --device "${DZ_SIM_DEVICE:-cpu}" --max-steps "${DZ_MAXSTEPS:-2000}" --stride "${DZ_STRIDE:-1}" )
[ -n "${DZ_H5:-}" ] && args+=( --h5 "$DZ_H5" )
# Drive the arms from an IK'd joint trajectory (eval_utils/export_xyzw_full.py output)
# instead of misreading the h5 EE poses as joint angles.
[ -n "${DZ_JOINTS_NPZ:-}" ] && args+=( --joints-npz "$DZ_JOINTS_NPZ" )
echo "==> apptainer ${args[*]}"
apptainer "${args[@]}"
echo "==> replay exit: $?"
