#!/bin/bash
# Run the h5 real-vs-sim replay inside the local isaac-lab Docker image (this workstation).
#
# Mirrors euler/replay_h5.sbatch but for a plain docker host with the official
# nvcr.io/nvidia/isaac-lab:2.3.1 image (Isaac Sim 4.5.0 + IsaacLab 2.3.1).
#
# Usage:
#   sudo bash docker/scripts/run_replay_sidebyside.sh [extra args to replay_h5_sidebyside.py]
# Examples:
#   sudo bash docker/scripts/run_replay_sidebyside.sh --max-steps 8          # quick plumbing check
#   sudo bash docker/scripts/run_replay_sidebyside.sh                        # full episode
set -uo pipefail

REPO="${DZ_REPO:-/home/wow/dreamzero}"
# Match the dreamzero target stack (Isaac Sim 4.5.0 + IsaacLab v2.1.0). The official
# isaac-lab:2.3.1 image bundles Isaac Sim 5.1 whose URDF importer API differs and breaks
# the URDF spawn. Build the repo image first: see the build command in this file's header.
IMAGE="${DZ_IMAGE:-dreamzero/isaac-sim:4.5.0}"
ISAACLAB_SH="${DZ_ISAACLAB_SH:-/opt/IsaacLab/isaaclab.sh}"   # /workspace/isaaclab/isaaclab.sh for the 2.3.1 image
OUT="${DZ_OUTPUT:-$REPO/output/sim}"
H5="${DZ_H5:-/app/20250826_111157.h5}"
DEVICE="${DZ_SIM_DEVICE:-cpu}"           # cpu physics + GPU rendering (euler-verified combo)
CACHE="${DZ_CACHE:-$HOME/.cache/dz_isaac}"

mkdir -p "$OUT" \
  "$CACHE/kit" "$CACHE/ov" "$CACHE/pip" "$CACHE/glcache" "$CACHE/computecache" \
  "$CACHE/ov_data" "$CACHE/logs" "$CACHE/documents"

exec docker run --rm --gpus all \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HOME=/root \
  -v "$REPO:/app" \
  -v "$OUT:/output" \
  -v "$CACHE/kit:/isaac-sim/kit/cache" \
  -v "$CACHE/ov:/root/.cache/ov" \
  -v "$CACHE/pip:/root/.cache/pip" \
  -v "$CACHE/glcache:/root/.cache/nvidia/GLCache" \
  -v "$CACHE/computecache:/root/.nv/ComputeCache" \
  -v "$CACHE/ov_data:/root/.local/share/ov/data" \
  -v "$CACHE/logs:/root/.nvidia-omniverse/logs" \
  -v "$CACHE/documents:/root/Documents" \
  -w /app \
  --entrypoint "$ISAACLAB_SH" \
  "$IMAGE" \
  -p eval_utils/replay_h5_sidebyside.py \
    --headless --enable_cameras --device "$DEVICE" --h5 "$H5" "$@"
