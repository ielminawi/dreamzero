"""Build the SIM-vs-train comparison PNG (login node, no GPU).

Layout: 2 rows (OAKD, ARIA). Each row: [GT train frame | NEW sim frame (post fixes)].
For OAKD also append the OLD-cfg sim candidate so the before/after is visible.
"""
import os
import cv2
import numpy as np

D = "/cluster/scratch/rjiang/dreamzero/output/cam_match"
STEP = int(os.environ.get("STEP", "1000"))
PW = 480  # per-panel width


def lab(im, t):
    im = im.copy()
    cv2.putText(im, t, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(im, t, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return im


def fit(path, label):
    im = cv2.imread(path)
    if im is None:
        im = np.full((360, PW, 3), 60, np.uint8)
        return lab(im, f"MISSING {label}")
    h, w = im.shape[:2]
    im = cv2.resize(im, (PW, int(h * PW / w)))
    return lab(im, label)


def pad_to(im, h):
    if im.shape[0] == h:
        return im
    out = np.zeros((h, im.shape[1], 3), np.uint8)
    out[: im.shape[0]] = im
    return out


# OAKD row: GT | NEW sim | OLD-cfg candidate (narrow focal=20 era == x19/x00 sweep frames
# were the wide matched lens; the true OLD render distortion is best shown by the narrow
# focal — use the closest archived narrow frame if present, else the wide sweep frame).
oakd_gt = fit(f"{D}/gt_oakd_{STEP}.png", f"GT TRAIN oakd {STEP}")
oakd_new = fit(f"{D}/match_oakd_step{STEP}.png", "SIM new (matched cam+dark hands)")
oakd_old = fit(f"{D}/sim_x00_z45_p26_step{STEP}.png", "SIM old (white hands, sweep frame)")

aria_gt = fit(f"{D}/gt_aria_{STEP}.png", f"GT TRAIN aria {STEP}")
aria_new = fit(f"{D}/match_aria_step{STEP}.png", "SIM new aria")

h = max(oakd_gt.shape[0], oakd_new.shape[0], oakd_old.shape[0],
        aria_gt.shape[0], aria_new.shape[0])
oakd_row = np.hstack([pad_to(x, h) for x in (oakd_gt, oakd_new, oakd_old)])
# pad aria row to the same width
aria_row = np.hstack([pad_to(x, h) for x in (aria_gt, aria_new)])
if aria_row.shape[1] < oakd_row.shape[1]:
    pad = np.zeros((h, oakd_row.shape[1] - aria_row.shape[1], 3), np.uint8)
    aria_row = np.hstack([aria_row, pad])

sheet = np.vstack([oakd_row, aria_row])
out = f"{D}/sim_vs_train_FIXED_step{STEP}.png"
cv2.imwrite(out, sheet)
print(f"wrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
