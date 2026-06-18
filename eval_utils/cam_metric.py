"""Score camera-sweep renders against GT by hand-blob centroid offset (login node, no GPU).

For each candidate and step: predict the left-arm flange pixel, segment the sim hand
(low-saturation bright blob near the prediction; the wall is out of view after the
backdrop fix) and the GT hand (largest dark component near the prediction after
morphological opening of the <70 gray mask), and report the centroid offset
(gt - sim). Offsets in px; positive u = GT right of sim, positive v = GT below sim.

Usage: .venv/bin/python eval_utils/cam_metric.py --cams output/cam_match/candidates_r4.json
"""
import argparse
import json
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, ".")

ap = argparse.ArgumentParser()
ap.add_argument("--cams", required=True)
ap.add_argument("--dir", default="output/cam_match")
ap.add_argument("--steps", type=int, nargs="+", default=[50, 1000, 1600, 2400])
ap.add_argument("--h5", default="/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/20250826_111157.h5")
args = ap.parse_args()

W, H, AP = 960, 540, 20.955
ARM_Y = 0.6127 / 2

with h5py.File(args.h5, "r") as f:
    Lw = f["observations/qpos_arm_left"][:, :3] + np.array([0, -ARM_Y, 0])


def cam_mats(c):
    eye = np.asarray(c["eye"], float)
    d = np.asarray(c["dir"], float); d /= np.linalg.norm(d)
    z = -d; up = np.array([0, 0, 1.0])
    x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    return eye, np.column_stack([x, y, z]), c["focal"] / AP * W


def proj(P, eye, R, fx):
    pc = R.T @ (P - eye)
    if pc[2] >= -0.05:
        return None
    return W / 2 + fx * pc[0] / (-pc[2]), H / 2 - fx * pc[1] / (-pc[2])


def blob_near(mask, near, r, min_area=1500):
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    best, bd = None, 1e9
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        d = np.hypot(cent[i][0] - near[0], cent[i][1] - near[1])
        if d < r and d < bd:
            best, bd = i, d
    if best is None:
        return None, 0
    return tuple(cent[best]), stats[best, cv2.CC_STAT_AREA]


gt = {st: cv2.imread(f"{args.dir}/gt_oakd_{st}.png") for st in args.steps}
cands = json.load(open(args.cams))

results = []
for c in cands:
    eye, R, fx = cam_mats(c)
    offs = []
    rows = []
    for st in args.steps:
        pr = proj(Lw[st], eye, R, fx)
        if pr is None or not (-100 < pr[0] < 1060 and -60 < pr[1] < 600):
            continue
        sim = cv2.imread(f"{args.dir}/sim_{c['name']}_step{st}.png")
        if sim is None:
            continue
        hsv = cv2.cvtColor(sim, cv2.COLOR_BGR2HSV)
        sc, sa = blob_near((hsv[..., 1] < 40) & (hsv[..., 2] > 120), pr, 200)
        if sc is None:
            continue
        g = cv2.cvtColor(gt[st], cv2.COLOR_BGR2GRAY)
        gc, ga = blob_near(g < 70, sc, 260, min_area=4000)
        if gc is None:
            continue
        offs.append((gc[0] - sc[0], gc[1] - sc[1]))
        rows.append(f"  step{st}: pred=({pr[0]:4.0f},{pr[1]:4.0f}) sim=({sc[0]:4.0f},{sc[1]:4.0f}) "
                    f"gt=({gc[0]:4.0f},{gc[1]:4.0f}) off=({gc[0]-sc[0]:+4.0f},{gc[1]-sc[1]:+4.0f})")
    if offs:
        m = np.mean(offs, axis=0)
        sd = np.std(offs, axis=0)
        score = float(np.mean([np.hypot(*o) for o in offs]))
        results.append((score, c["name"], m, sd, rows))

results.sort()
for score, name, m, sd, rows in results:
    print(f"{name:16s} score={score:6.1f}px  mean=({m[0]:+5.1f},{m[1]:+5.1f})  sd=({sd[0]:4.1f},{sd[1]:4.1f})")
    for r in rows:
        print(r)
