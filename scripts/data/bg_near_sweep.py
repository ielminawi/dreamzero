"""
Sweep the 'near' radius: keep a halo of table around arms/objects (so the dark Orca hand is
visible on it) and remove floor/tiles farther out. Smaller radius -> removes MORE.
cols: original | near 0.05 | near 0.09 | near 0.14 | remove   (black fill)
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import cv2, h5py, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bg_removal  # noqa: E402

RAW = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries"
OUT = Path("/cluster/scratch/rjiang/dreamzero/output/bg_near_sweep"); OUT.mkdir(parents=True, exist_ok=True)
TARGET = (640, 480)


def grab():
    files = sorted(glob.glob(f"{RAW}/*.h5"))
    picks = [(files[0], "aria_rgb_cam", 0.30), (files[0], "aria_rgb_cam", 0.50),
             (files[len(files)//2], "aria_rgb_cam", 0.45), (files[0], "oakd_front_view", 0.45),
             (files[len(files)//2], "oakd_front_view", 0.55), (files[-1], "aria_rgb_cam", 0.5)]
    out = []
    for fp, cam, fr in picks:
        with h5py.File(fp, "r") as f:
            ds = f[f"observations/images/{cam}/color"]; T = ds.shape[0]
            out.append(cv2.resize(ds[int(T*fr)], TARGET, interpolation=cv2.INTER_AREA))
    return np.stack(out)


def main():
    frames = grab()
    br = bg_removal.BackgroundRemover(backend="mask2former", fill=0)
    variants = [("near", 0.05), ("near", 0.09), ("near", 0.14), ("remove", None)]
    outs = {"original": frames}
    for name, fr in variants:
        br.floor_mode = name
        if fr is not None:
            br.near_frac = fr
        outs[(name, fr)] = br.apply(frames)
        print("done", name, fr, flush=True)
    order = ["original"] + [(n, f) for n, f in variants]
    labels = ["original", "near .05", "near .09", "near .14", "remove"]
    rows = [np.concatenate([cv2.resize(outs[k][i], (230, 173)) for k in order], 1) for i in range(len(frames))]
    sheet = np.concatenate(rows, 0)
    cv2.imwrite(str(OUT / "contact.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("wrote", OUT / "contact.png", "| cols:", " | ".join(labels), flush=True)


if __name__ == "__main__":
    main()
