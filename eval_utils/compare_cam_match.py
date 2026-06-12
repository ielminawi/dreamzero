"""Build comparison sheets from match_oakd_camera.py output (runs on the login node, no GPU).

For each step: one contact sheet with a row per candidate: [GT | SIM | 50/50 blend].
The blend makes arm misalignment obvious (ghosting). Usage:
  .venv/bin/python eval_utils/compare_cam_match.py [--dir output/cam_match] [--steps 50 1000 1600]
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="output/cam_match")
ap.add_argument("--steps", type=int, nargs="+", default=[50, 1000, 1600, 2400])
ap.add_argument("--width", type=int, default=480, help="per-panel width in the sheet")
args = ap.parse_args()


def label(im, text):
    im = im.copy()
    cv2.putText(im, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(im, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return im


for st in args.steps:
    gt_path = os.path.join(args.dir, f"gt_oakd_{st}.png")
    if not os.path.exists(gt_path):
        print(f"skip step {st}: no {gt_path}")
        continue
    gt = cv2.imread(gt_path)
    h, w = gt.shape[:2]
    pw = args.width
    ph = int(h * pw / w)
    gt_s = cv2.resize(gt, (pw, ph))
    rows = []
    sims = sorted(glob.glob(os.path.join(args.dir, f"sim_*_step{st}.png")))
    for sp in sims:
        name = re.match(rf"sim_(.+)_step{st}\.png", os.path.basename(sp)).group(1)
        sim = cv2.imread(sp)
        if sim.shape[:2] != (h, w):
            sim = cv2.resize(sim, (w, h))
        sim_s = cv2.resize(sim, (pw, ph))
        blend = cv2.addWeighted(gt_s, 0.5, sim_s, 0.5, 0)
        row = np.hstack([label(gt_s, f"GT {st}"), label(sim_s, name), label(blend, "blend")])
        cv2.imwrite(os.path.join(args.dir, f"row_{name}_step{st}.png"), row)
        rows.append(row)
    if not rows:
        print(f"skip step {st}: no sim frames")
        continue
    sheet = np.vstack(rows)
    out = os.path.join(args.dir, f"sheet_step{st}.png")
    cv2.imwrite(out, sheet)
    print(f"wrote {out} ({len(rows)} candidates)")
