#!/bin/bash
#SBATCH --job-name=dz_gear
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/gear.%j.log

# Run ONCE after the preprocess array succeeds. Writes meta/info.json, fixes the
# global frame index, then builds GEAR metadata (modality/stats/relative/episodes/
# tasks/embodiment), then validates the dataset is decord-loadable. CPU job.
set -uo pipefail
cd /cluster/scratch/rjiang/dreamzero

RAW=${RAW:-/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries}
DATA_ROOT=${DATA_ROOT:-/cluster/scratch/rjiang/dreamzero/data/franka_orca_bag_groceries_lerobot}

echo "==> finalize info.json + global index"
.venv/bin/python scripts/data/convert_bag_groceries.py \
  --input-dir "$RAW" --output-dir "$DATA_ROOT" \
  --fps 50 --target-resolution 640x480 --finalize || exit 1

echo "==> build GEAR metadata (modality/stats/relative/episodes/tasks/embodiment)"
.venv/bin/python scripts/data/convert_lerobot_to_gear.py \
  --dataset-path "$DATA_ROOT" \
  --embodiment-tag franka_orca_bimanual \
  --state-keys  '{"left_arm_joint_pos":[0,7],"right_arm_joint_pos":[7,14],"left_hand_joint_pos":[14,31],"right_hand_joint_pos":[31,48]}' \
  --action-keys '{"left_arm_joint_pos":[0,7],"right_arm_joint_pos":[7,14],"left_hand_joint_pos":[14,31],"right_hand_joint_pos":[31,48]}' \
  --relative-action-keys left_arm_joint_pos right_arm_joint_pos left_hand_joint_pos right_hand_joint_pos \
  --task-key annotation.task --action-horizon 24 --fps 50 --force || exit 1

echo "==> validate (decord + parquet + meta keys vs training config)"
.venv/bin/python - "$DATA_ROOT" <<'PY' || exit 1
import sys, json, glob, numpy as np, pandas as pd
root = sys.argv[1]
info = json.load(open(f"{root}/meta/info.json"))
mod  = json.load(open(f"{root}/meta/modality.json"))
assert info["fps"] == 50, info["fps"]
assert info["features"]["observation.state"]["shape"] == [48]
assert info["features"]["action"]["shape"] == [48]
vids = [k for k,v in info["features"].items() if v["dtype"]=="video"]
assert set(vids) == {"observation.images.aria_rgb_cam","observation.images.oakd_front_view"}, vids
for k in ["left_arm_joint_pos","right_arm_joint_pos","left_hand_joint_pos","right_hand_joint_pos"]:
    assert k in mod["state"] and k in mod["action"], k
assert set(mod["video"]) == {"aria_rgb_cam","oakd_front_view"}, list(mod["video"])
for f in ["stats.json","relative_stats_dreamzero.json","episodes.jsonl","tasks.jsonl","embodiment.json"]:
    assert glob.glob(f"{root}/meta/{f}"), f
# decord load one episode's videos and check frame count == parquet length
import decord
pq = sorted(glob.glob(f"{root}/data/chunk-000/episode_000000.parquet"))[0]
df = pd.read_parquet(pq); T = len(df)
assert len(df["observation.state"].iloc[0]) == 48 and len(df["action"].iloc[0]) == 48
for cam in ["aria_rgb_cam","oakd_front_view"]:
    vr = decord.VideoReader(f"{root}/videos/chunk-000/observation.images.{cam}/episode_000000.mp4")
    fr = vr[0].asnumpy()
    assert abs(len(vr)-T) <= 2, (cam, len(vr), T)
    assert fr.shape == (480,640,3), (cam, fr.shape)
print(f"VALIDATION OK: {info['total_episodes']} eps, {info['total_frames']} frames, "
      f"ep0 T={T}, state/action=48, cams={sorted(mod['video'])}, decord readable.")
PY
echo "==> GEAR + validation done. Dataset ready at: $DATA_ROOT"
