"""
Diagnose the dark-Orca-hand visibility problem in the aria view.
cols: original | remove:green(removed=green) | remove:black | remove:gray | near:black
If the hand is GREEN in col2 -> segmentation is deleting it (fix the mask).
If the hand is invisible in col3(black) but visible in col4(gray) -> it's just a dark hand on
black (contrast problem -> lighter fill or brighten foreground).
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import cv2, h5py, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bg_removal  # noqa: E402

RAW = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries"
OUT = Path("/cluster/scratch/rjiang/dreamzero/output/bg_hand"); OUT.mkdir(parents=True, exist_ok=True)
TARGET = (640, 480); GREEN = np.array([0, 255, 0], np.float32)


def grab():
    files = sorted(glob.glob(f"{RAW}/*.h5"))
    picks = [(files[0], 0.30), (files[0], 0.50), (files[0], 0.70),
             (files[len(files)//2], 0.45), (files[-1], 0.5)]
    out = []
    for fp, fr in picks:
        with h5py.File(fp, "r") as f:
            ds = f["observations/images/aria_rgb_cam/color"]; T = ds.shape[0]
            out.append(cv2.resize(ds[int(T*fr)], TARGET, interpolation=cv2.INTER_AREA))
    return np.stack(out)


def main():
    frames = grab()
    br = bg_removal.BackgroundRemover(backend="mask2former", fill=0)
    br.floor_mode = "remove"; a_rm = br.foreground_alpha(frames)[..., None]
    br.floor_mode = "near";   a_nr = br.foreground_alpha(frames)[..., None]
    rows = []
    for i in range(len(frames)):
        f = frames[i].astype(np.float32)
        rm_green = (f * a_rm[i] + GREEN * (1 - a_rm[i])).astype(np.uint8)
        rm_black = (f * a_rm[i]).astype(np.uint8)
        rm_gray = (f * a_rm[i] + 128 * (1 - a_rm[i])).astype(np.uint8)
        nr_black = (f * a_nr[i]).astype(np.uint8)
        cols = [frames[i], rm_green, rm_black, rm_gray, nr_black]
        rows.append(np.concatenate([cv2.resize(x, (256, 192)) for x in cols], 1))
    sheet = np.concatenate(rows, 0)
    cv2.imwrite(str(OUT / "contact.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("wrote", OUT / "contact.png", "| cols: original | remove:green | remove:black | remove:gray | near:black", flush=True)


if __name__ == "__main__":
    main()
