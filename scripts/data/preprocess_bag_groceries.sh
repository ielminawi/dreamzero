#!/bin/bash
# Preprocess bag_groceries raw HDF5 -> LeRobot v2 -> GEAR/DreamZero metadata
# for the franka_orca_bimanual embodiment (state 48, action 48, 2 cameras).
# Two idempotent stages. Parameterise via env vars (see defaults below).
set -euo pipefail

REPO=/cluster/scratch/rjiang/dreamzero
PY="$REPO/.venv/bin/python"

SRC="${SRC:-/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries}"
# ARM_TO_JOINTS=1: IK the arm EE-pose 7-vectors into Panda joint angles during conversion
# (true joint-space dataset; see docs/EE_POSE_AND_IK_PIPELINE.md). Defaults to a separate
# output dir so the EE-space dataset is not clobbered.
ARM_TO_JOINTS="${ARM_TO_JOINTS:-0}"
if [ "$ARM_TO_JOINTS" = "1" ]; then
  DST="${DST:-$REPO/data/franka_orca_lerobot_joints}"
else
  DST="${DST:-$REPO/data/franka_orca_lerobot}"
fi
FPS="${FPS:-50}"
TASK="${TASK:-bag the groceries}"
TARGET_RES="${TARGET_RES:-640x480}"        # both cameras -> single resolution (matches eval client)
NW="${NW:-14}"
ACTION_HORIZON="${ACTION_HORIZON:-24}"
MAX_EPISODES="${MAX_EPISODES:-}"           # empty = all episodes

# state/action are the concatenation [arm_left(7), arm_right(7), hand_left(17), hand_right(17)] = 48
KEYS='{"left_arm_joint_pos":[0,7],"right_arm_joint_pos":[7,14],"left_hand_joint_pos":[14,31],"right_hand_joint_pos":[31,48]}'

cd "$REPO"
echo "[preprocess] SRC=$SRC"
echo "[preprocess] DST=$DST  fps=$FPS  res=$TARGET_RES  workers=$NW  max=${MAX_EPISODES:-ALL}"

MAXARG=()
[ -n "$MAX_EPISODES" ] && MAXARG=(--max-episodes "$MAX_EPISODES")
JOINTARG=()
[ "$ARM_TO_JOINTS" = "1" ] && JOINTARG=(--arm-ee-to-joints) && echo "[preprocess] ARM EE->JOINT conversion ON"

echo "[stage 1/2] HDF5 -> LeRobot v2 (parquet + mp4)"
"$PY" scripts/data/convert_h5_to_lerobot.py \
  --input-dir "$SRC" \
  --output-dir "$DST" \
  --fps "$FPS" \
  --task "$TASK" \
  --target-resolution "$TARGET_RES" \
  --num-workers "$NW" \
  "${MAXARG[@]}" \
  "${JOINTARG[@]}"

echo "[stage 2/2] LeRobot v2 -> GEAR metadata (modality/stats/relative/episodes/tasks)"
"$PY" scripts/data/convert_lerobot_to_gear.py \
  --dataset-path "$DST" \
  --embodiment-tag franka_orca_bimanual \
  --state-keys  "$KEYS" \
  --action-keys "$KEYS" \
  --relative-action-keys left_arm_joint_pos right_arm_joint_pos left_hand_joint_pos right_hand_joint_pos \
  --task-key annotation.task \
  --fps "$FPS" \
  --action-horizon "$ACTION_HORIZON" \
  --force

echo "[preprocess] DONE -> $DST"
