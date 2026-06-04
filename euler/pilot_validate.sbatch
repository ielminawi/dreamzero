#!/bin/bash
#SBATCH --job-name=dz_pilot
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/pilot.%j.log

# Mechanical end-to-end validation of the NEW converter + GEAR + decord contract on 2
# real episodes, NO background removal (so it does not pre-empt the bg review). CPU only.
set -uo pipefail
cd /cluster/scratch/rjiang/dreamzero
RAW=/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries
PILOT=/cluster/scratch/rjiang/dreamzero/data/_pilot_validate
rm -rf "$PILOT"

echo "==> convert 2 episodes (no bg)"
.venv/bin/python scripts/data/convert_bag_groceries.py \
  --input-dir "$RAW" --output-dir "$PILOT" --fps 50 --target-resolution 640x480 \
  --max-episodes 2 --num-shards 1 --shard-index 0 --bg-removal none || exit 1

echo "==> finalize"
.venv/bin/python scripts/data/convert_bag_groceries.py \
  --input-dir "$RAW" --output-dir "$PILOT" --fps 50 --target-resolution 640x480 \
  --max-episodes 2 --finalize || exit 1

echo "==> GEAR"
.venv/bin/python scripts/data/convert_lerobot_to_gear.py \
  --dataset-path "$PILOT" --embodiment-tag franka_orca_bimanual \
  --state-keys  '{"left_arm_joint_pos":[0,7],"right_arm_joint_pos":[7,14],"left_hand_joint_pos":[14,31],"right_hand_joint_pos":[31,48]}' \
  --action-keys '{"left_arm_joint_pos":[0,7],"right_arm_joint_pos":[7,14],"left_hand_joint_pos":[14,31],"right_hand_joint_pos":[31,48]}' \
  --relative-action-keys left_arm_joint_pos right_arm_joint_pos left_hand_joint_pos right_hand_joint_pos \
  --task-key annotation.task --action-horizon 24 --fps 50 --force || exit 1

echo "==> validate (decord + parquet + meta vs training config)"
.venv/bin/python - "$PILOT" <<'PY' || exit 1
import sys, json, glob, pandas as pd, decord
root=sys.argv[1]
info=json.load(open(f"{root}/meta/info.json")); mod=json.load(open(f"{root}/meta/modality.json"))
assert info["fps"]==50 and info["features"]["observation.state"]["shape"]==[48] and info["features"]["action"]["shape"]==[48]
assert set(mod["video"])=={"aria_rgb_cam","oakd_front_view"}
for k in ["left_arm_joint_pos","right_arm_joint_pos","left_hand_joint_pos","right_hand_joint_pos"]:
    assert k in mod["state"] and k in mod["action"], k
for f in ["stats.json","relative_stats_dreamzero.json","episodes.jsonl","tasks.jsonl","embodiment.json"]:
    assert glob.glob(f"{root}/meta/{f}"), f
df=pd.read_parquet(f"{root}/data/chunk-000/episode_000000.parquet"); T=len(df)
assert len(df['observation.state'].iloc[0])==48 and len(df['action'].iloc[0])==48
for cam in ["aria_rgb_cam","oakd_front_view"]:
    vr=decord.VideoReader(f"{root}/videos/chunk-000/observation.images.{cam}/episode_000000.mp4")
    assert abs(len(vr)-T)<=2 and vr[0].asnumpy().shape==(480,640,3), (cam,len(vr),T,vr[0].asnumpy().shape)
import numpy as np
rs=json.load(open(f"{root}/meta/relative_stats_dreamzero.json"))
assert len(rs["left_arm_joint_pos"]["q99"])==7 and len(rs["left_hand_joint_pos"]["q99"])==17
print(f"PILOT VALIDATION OK: {info['total_episodes']} eps, {info['total_frames']} frames, ep0 T={T}, "
      f"state/action=48, rel-stats dims ok, decord readable {sorted(mod['video'])}.")
PY
echo "==> pilot done; cleaning up pilot dataset"
du -sh "$PILOT" 2>/dev/null
rm -rf "$PILOT"
echo "==> PILOT PASS"
