"""
Show the three floor-handling modes side by side so a human can pick the right tradeoff.
Output: output/bg_modes/contact.png  rows=frames; cols = original | keep | near | remove  (all black fill)
  keep   = remove only walls (table+floor+tiles all stay)         [safest, least bg removed]
  near   = keep table near arms/objects, drop far floor/tiles+walls [best 'remove tiles keep table']
  remove = drop all floor+walls (table often lost)                 [most bg removed]
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import cv2, h5py, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bg_removal  # noqa: E402

RAW = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries"
OUT = Path("/cluster/scratch/rjiang/dreamzero/output/bg_modes"); OUT.mkdir(parents=True, exist_ok=True)
TARGET = (640, 480)


def grab():
    files = sorted(glob.glob(f"{RAW}/*.h5"))
    picks = [(files[0], "aria_rgb_cam", 0.45), (files[0], "oakd_front_view", 0.45),
             (files[len(files)//2], "aria_rgb_cam", 0.4), (files[len(files)//2], "oakd_front_view", 0.55),
             (files[-1], "aria_rgb_cam", 0.5), (files[-1], "oakd_front_view", 0.5)]
    out = []
    for fp, cam, fr in picks:
        with h5py.File(fp, "r") as f:
            ds = f[f"observations/images/{cam}/color"]; T = ds.shape[0]
            out.append(cv2.resize(ds[int(T*fr)], TARGET, interpolation=cv2.INTER_AREA))
    return np.stack(out)


def main():
    frames = grab()
    br = bg_removal.BackgroundRemover(backend="mask2former", fill=0)
    print("device", br.device, "model", br.model_name, flush=True)
    outs = {"original": frames}
    for mode in ["keep", "near", "remove"]:
        br.floor_mode = mode
        outs[mode] = br.apply(frames)
        print("done mode", mode, flush=True)
    order = ["original", "keep", "near", "remove"]
    rows = [np.concatenate([cv2.resize(outs[k][i], (220, 165)) for k in order], 1) for i in range(len(frames))]
    sheet = np.concatenate(rows, 0)
    cv2.imwrite(str(OUT / "contact.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("wrote", OUT / "contact.png", sheet.shape, "| cols: original | keep | near | remove", flush=True)


if __name__ == "__main__":
    main()
