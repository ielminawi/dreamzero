#!/bin/bash
#SBATCH --job-name=dz_cammatch
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:24g
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/cammatch.%j.log

set -uo pipefail
module load eth_proxy 2>/dev/null || true
source /cluster/scratch/rjiang/dreamzero/euler/scripts/_env.sh
mkdir -p "$DZ_OVHOME" "$DZ_LOGDIR" "$DZ_REPO/output/cam_match"
proxy="${DZ_PROXY:-http://proxy.service.consul:3128}"
args=( run --nv --env TERM=xterm
       --env "http_proxy=$proxy" --env "https_proxy=$proxy"
       --env "HTTP_PROXY=$proxy" --env "HTTPS_PROXY=$proxy"
       --env "no_proxy=localhost,127.0.0.1" --env "NO_PROXY=localhost,127.0.0.1"
       --home "$DZ_OVHOME:$DZ_OVHOME"
       --bind "$DZ_REPO:/app" --bind "$DZ_REPO/output/cam_match:/output"
       --bind /cluster/work/cvg/data/Egoverse:/cluster/work/cvg/data/Egoverse:ro
       --pwd /app )
[ -f "$DZ_OVERLAY" ] && args+=( --overlay "$DZ_OVERLAY" )
args+=( "$DZ_SIF" eval_utils/match_oakd_camera.py --headless --enable_cameras
        --device "${DZ_SIM_DEVICE:-cpu}"
        --cams "/app/output/cam_match/${DZ_CAM_JSON:-candidates.json}" )
echo "==> apptainer ${args[*]}"
apptainer "${args[@]}"
echo "==> match_oakd_camera exit: $?"
