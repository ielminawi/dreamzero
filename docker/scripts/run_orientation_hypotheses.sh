#!/bin/bash
# Freeze the bimanual rig at one recorded frame and render multi-view high-res snapshots
# (two robot cameras + 4 free overview cameras + real|sim comparisons), inside the local
# Isaac docker image. Same stack/mounts as run_replay_sidebyside.sh.
#
# Usage:
#   sudo bash docker/scripts/run_orientation_hypotheses.sh [extra args to snapshot_frame0_multiview.py]
# Examples:
#   sudo bash docker/scripts/run_orientation_hypotheses.sh                 # frame 0
#   sudo bash docker/scripts/run_orientation_hypotheses.sh --frame 250     # a later frame
set -uo pipefail

REPO="${DZ_REPO:-/home/wow/dreamzero}"
IMAGE="${DZ_IMAGE:-dreamzero/isaac-sim:4.5.0}"
ISAACLAB_SH="${DZ_ISAACLAB_SH:-/opt/IsaacLab/isaaclab.sh}"
OUT="${DZ_OUTPUT:-$REPO/output/sim}"
H5="${DZ_H5:-/app/20250826_111157.h5}"
DEVICE="${DZ_SIM_DEVICE:-cpu}"           # cpu physics + GPU rendering
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
  -p eval_utils/render_orientation_hypotheses.py \
    --headless --enable_cameras --device "$DEVICE" --h5 "$H5" "$@"
