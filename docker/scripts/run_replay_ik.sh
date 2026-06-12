#!/bin/bash
# Run the IK replay (eval_utils/replay_h5_ik.py) inside the local isaac-lab Docker
# image. Same stack/mounts as run_replay_sidebyside.sh, but drives the articulation
# by IK to the recorded EE poses instead of joint angles.
#
# Usage:
#   sudo bash docker/scripts/run_replay_ik.sh [extra args to replay_h5_ik.py]
# Hypothesis check (EE pose == HAND): target the palm to the RAW recorded pose,
# no quaternion correction, two well-separated timesteps:
#   sudo bash docker/scripts/run_replay_ik.sh --ee-body palm --stride 8 --max-steps 200 --settle-steps 3
set -uo pipefail

REPO="${DZ_REPO:-/home/wow/dreamzero}"
IMAGE="${DZ_IMAGE:-dreamzero/isaac-sim:4.5.0}"
ISAACLAB_SH="${DZ_ISAACLAB_SH:-/opt/IsaacLab/isaaclab.sh}"
OUT="${DZ_OUTPUT:-$REPO/output/ik_replay}"
H5="${DZ_H5:-/app/20250826_111157.h5}"
DEVICE="${DZ_SIM_DEVICE:-cpu}"
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
  -p eval_utils/replay_h5_ik.py \
    --headless --enable_cameras --device "$DEVICE" --h5 "$H5" --out-dir /output "$@"
