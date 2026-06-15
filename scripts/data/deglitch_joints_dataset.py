#!/usr/bin/env python3
"""Build a CLEANED joint-space dataset: repair physically-impossible glitch frames in
state+action (median-filter outlier detection -> linear interpolation), then recompute
the relative-action normalization stats on the cleaned data with robust quantiles.

Why: the raw joints data has corrupt frames (per-frame joint jumps up to ~4 rad/frame =
~200 rad/s at 50fps, impossible for a Panda; present in ~half the episodes). These blow up
the relative-action q99 spans (×15-42 vs real motion) so GT normalized targets become ~0.02
and the flow-matching action head can't resolve them -> over-shoots -> loses to stay-still.

Cleaning removes the corrupt extremes; robust q025/q975 quantiles then handle the remaining
legit-but-rare fast-transport chunks. Videos are observations (untouched) -> symlinked.

Usage: python scripts/data/deglitch_joints_dataset.py [SRC] [DST]
"""
import json, sys, os, shutil, glob
import numpy as np, pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/franka_orca_lerobot_joints"
DST = sys.argv[2] if len(sys.argv) > 2 else "data/franka_orca_lerobot_joints_clean"
H = 24
# residual-from-local-median threshold per dim group (rad). Real motion at 50fps stays well
# under these (arm p99 ~0.03, p99.9 ~0.3 is already glitch); set above real, below glitch.
THR = np.zeros(48); THR[0:14] = 0.12; THR[14:48] = 0.30   # arms / hands
RK = ["left_arm_joint_pos", "right_arm_joint_pos", "left_hand_joint_pos", "right_hand_joint_pos"]

def deglitch(arr):
    """arr (N,48) -> cleaned copy + #cells repaired. Median(win5) outlier -> linear interp."""
    N = arr.shape[0]; out = arr.copy()
    med = pd.DataFrame(arr).rolling(5, center=True, min_periods=1).median().values
    bad = np.abs(arr - med) > THR[None, :]
    nbad = 0
    idx = np.arange(N)
    for d in range(48):
        bd = bad[:, d]
        if bd.any() and (~bd).sum() >= 2:
            out[bd, d] = np.interp(idx[bd], idx[~bd], arr[~bd, d])
            nbad += int(bd.sum())
    return out, nbad

mod = json.load(open(f"{SRC}/meta/modality.json")); akeys = mod["action"]; skeys = mod["state"]
files = sorted(glob.glob(f"{SRC}/data/chunk-000/episode_*.parquet"))
os.makedirs(f"{DST}/data/chunk-000", exist_ok=True)

tot_s = tot_a = 0; eps_touched = 0
pool = {k: [] for k in RK}
for fp in files:
    df = pd.read_parquet(fp)
    st = np.stack(df["observation.state"].values).astype(np.float64)
    ac = np.stack(df["action"].values).astype(np.float64)
    st2, nb1 = deglitch(st); ac2, nb2 = deglitch(ac)
    tot_s += nb1; tot_a += nb2; eps_touched += int((nb1 + nb2) > 0)
    df["observation.state"] = list(st2.astype(np.float32))
    df["action"] = list(ac2.astype(np.float32))
    df.to_parquet(f"{DST}/data/chunk-000/{os.path.basename(fp)}")
    # accumulate cleaned chunk-relative deltas for robust stats (subsample chunk starts)
    N = len(df)
    for k in RK:
        a0, a1 = akeys[k]["start"], akeys[k]["end"]; s0, s1 = skeys[k]["start"], skeys[k]["end"]
        for i in range(0, max(N - H, 0), 3):
            pool[k].append(ac2[i:i+H, a0:a1] - st2[i, s0:s1])

print(f"episodes={len(files)}  repaired cells: state={tot_s} action={tot_a}  episodes_touched={eps_touched}")

# robust relative-action stats (q025/q975 written into q01/q99 fields the normalizer reads)
old = json.load(open(f"{SRC}/meta/relative_stats_dreamzero.json"))
new = {}
print(f"\n{'key':22s} {'OLD span/2':>10s} {'CLEAN q01/99':>13s} {'CLEAN q025/975':>15s}")
for k in RK:
    d = np.concatenate(pool[k], 0)
    q1, q99 = np.quantile(d, 0.01, 0), np.quantile(d, 0.99, 0)
    q25, q975 = np.quantile(d, 0.025, 0), np.quantile(d, 0.975, 0)
    new[k] = {"max": np.max(d, 0).tolist(), "min": np.min(d, 0).tolist(),
              "mean": np.mean(d, 0).tolist(), "std": np.std(d, 0).tolist(),
              "q01": q25.tolist(), "q99": q975.tolist()}   # robust q025/q975 into q01/q99 fields
    oh = ((np.array(old[k]["q99"]) - np.array(old[k]["q01"])) / 2).mean()
    print(f"{k:22s} {oh:10.3f} {((q99-q1)/2).mean():13.3f} {((q975-q25)/2).mean():15.3f}")

# assemble variant: symlink videos, copy meta, write cleaned relative stats
for sub in ["videos"]:
    link = os.path.join(DST, sub)
    if os.path.islink(link): os.unlink(link)
    os.symlink(os.path.abspath(os.path.join(SRC, sub)), link)
mdst = os.path.join(DST, "meta")
if os.path.exists(mdst): shutil.rmtree(mdst)
shutil.copytree(os.path.join(SRC, "meta"), mdst)
json.dump(new, open(os.path.join(mdst, "relative_stats_dreamzero.json"), "w"), indent=2)
print(f"\nCleaned dataset at {DST}  (videos symlinked, parquets rewritten, robust stats)")
print("dims:", [len(new[k]["q99"]) for k in RK])
