#!/usr/bin/env python
"""Validate a converted LeRobot+GEAR dataset for franka_orca_bimanual training.

Checks meta files, modality.json structure/ranges, parquet shapes, decord video
readability (the training video_backend), and stats dims. Exits non-zero on any
failure so it can gate SLURM job dependencies (afterok).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECT_KEYS = {
    "left_arm_joint_pos": [0, 7],
    "right_arm_joint_pos": [7, 14],
    "left_hand_joint_pos": [14, 31],
    "right_hand_joint_pos": [31, 48],
}
EXPECT_REL_DIMS = {"left_arm_joint_pos": 7, "right_arm_joint_pos": 7,
                   "left_hand_joint_pos": 17, "right_hand_joint_pos": 17}
EXPECT_VIDEO = ["aria_rgb_cam", "oakd_front_view"]
STATE_DIM = ACTION_DIM = 48


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def ok(msg):
    print("  ok:", msg)


def main():
    if len(sys.argv) < 2:
        fail("usage: validate_lerobot_dataset.py <dataset_dir>")
    root = Path(sys.argv[1])
    meta = root / "meta"

    for f in ["info.json", "modality.json", "embodiment.json", "stats.json",
              "relative_stats_dreamzero.json", "tasks.jsonl", "episodes.jsonl"]:
        if not (meta / f).exists():
            fail(f"missing meta/{f}")
    ok("all 7 meta files present")

    info = json.load(open(meta / "info.json"))
    modality = json.load(open(meta / "modality.json"))

    for sect in ("state", "action"):
        keys = modality.get(sect, {})
        if set(keys) != set(EXPECT_KEYS):
            fail(f"modality.{sect} keys {sorted(keys)} != {sorted(EXPECT_KEYS)}")
        for k, (s, e) in EXPECT_KEYS.items():
            if [keys[k]["start"], keys[k]["end"]] != [s, e]:
                fail(f"modality.{sect}.{k} range [{keys[k]['start']},{keys[k]['end']}] != [{s},{e}]")
        ok(f"modality.{sect}: 4 keys, ranges sum to {ACTION_DIM} dims")

    if set(modality.get("video", {})) != set(EXPECT_VIDEO):
        fail(f"modality.video {sorted(modality.get('video', {}))} != {EXPECT_VIDEO}")
    ok("modality.video: aria_rgb_cam + oakd_front_view")
    if "task" not in modality.get("annotation", {}):
        fail("modality.annotation.task missing")
    ok("modality.annotation.task present")

    ep0 = root / "data" / "chunk-000" / "episode_000000.parquet"
    if not ep0.exists():
        fail(f"missing {ep0}")
    df = pd.read_parquet(ep0)
    for col in ["observation.state", "action", "annotation.task", "timestamp"]:
        if col not in df.columns:
            fail(f"parquet missing column {col}")
    s0 = np.asarray(df["observation.state"].iloc[0])
    a0 = np.asarray(df["action"].iloc[0])
    if s0.shape[-1] != STATE_DIM:
        fail(f"observation.state dim {s0.shape} != {STATE_DIM}")
    if a0.shape[-1] != ACTION_DIM:
        fail(f"action dim {a0.shape} != {ACTION_DIM}")
    T = len(df)
    ok(f"parquet ep0: {T} rows, state {STATE_DIM}d, action {ACTION_DIM}d, "
       f"task='{df['annotation.task'].iloc[0]}'")

    import decord
    for cam in EXPECT_VIDEO:
        vp = root / "videos" / "chunk-000" / f"observation.images.{cam}" / "episode_000000.mp4"
        if not vp.exists():
            fail(f"missing video {vp}")
        vr = decord.VideoReader(str(vp))
        n = len(vr)
        fr = vr[0].asnumpy()
        ok(f"video {cam}: decord read {n} frames, frame shape {tuple(fr.shape)}")
        if n != T:
            print(f"  WARN: {cam} frame count {n} != parquet rows {T}")

    stats = json.load(open(meta / "stats.json"))
    for col, dim in [("observation.state", STATE_DIM), ("action", ACTION_DIM)]:
        if col not in stats:
            fail(f"stats.json missing {col}")
        if len(stats[col]["q99"]) != dim:
            fail(f"stats.json {col} q99 dim {len(stats[col]['q99'])} != {dim}")
    ok("stats.json: observation.state + action (q01/q99) present")

    rel = json.load(open(meta / "relative_stats_dreamzero.json"))
    for k, d in EXPECT_REL_DIMS.items():
        if k not in rel:
            fail(f"relative_stats_dreamzero.json missing {k}")
        if len(rel[k]["q99"]) != d:
            fail(f"relative_stats {k} dim {len(rel[k]['q99'])} != {d}")
    ok("relative_stats_dreamzero.json: 4 keys with 7/7/17/17 dims")

    print(f"\nVALIDATION PASSED  ({root})  episodes={info.get('total_episodes')} fps={info.get('fps')}")


if __name__ == "__main__":
    main()
