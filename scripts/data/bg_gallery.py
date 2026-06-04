"""
Build a before/after gallery for background removal so a human can eyeball quality
before committing to full preprocessing. Produces:
  output/bg_gallery/contact.png   - grid of diverse frames, original (top) vs masked (bottom)
  output/bg_gallery/clip_<cam>.mp4 - one episode segment, original|masked side-by-side (temporal stability)
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bg_removal  # noqa: E402

RAW = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries"
OUT = Path("/cluster/scratch/rjiang/dreamzero/output/bg_gallery")
TARGET = (640, 480)  # W,H — same resize as the converter
CAMS = ["aria_rgb_cam", "oakd_front_view"]


def resize(img):
    return cv2.resize(img, TARGET, interpolation=cv2.INTER_AREA)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{RAW}/*.h5"))
    br = bg_removal.BackgroundRemover(batch_size=8)  # defaults: mask2former, floor_mode=remove, gray fill
    print("device", br.device, "fill", br.fill)

    # ---- contact sheet: 4 episodes x 2 cams x ~2 timepoints ----
    pick = [files[0], files[len(files) // 3], files[2 * len(files) // 3], files[-1]]
    tiles = []
    for fp in pick:
        with h5py.File(fp, "r") as f:
            T = f["observations/qpos_arm_left"].shape[0]
            for cam in CAMS:
                ds = f[f"observations/images/{cam}/color"]
                for frac in (0.35, 0.7):
                    t = int(T * frac)
                    orig = resize(ds[t])
                    masked = br.apply(orig[None])[0]
                    pair = np.concatenate([orig, masked], axis=0)  # 480 -> 960 tall
                    pair = cv2.resize(pair, (240, 360))
                    tiles.append(pair)
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    cell_h, cell_w = tiles[0].shape[:2]
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = t
    cv2.imwrite(str(OUT / "contact.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("wrote contact.png", sheet.shape)

    # ---- temporal clip: one episode, every 4th frame, original|masked ----
    with h5py.File(pick[0], "r") as f:
        T = f["observations/qpos_arm_left"].shape[0]
        seg = range(int(T * 0.3), int(T * 0.3) + 480, 4)  # ~120 frames
        for cam in CAMS:
            ds = f[f"observations/images/{cam}/color"]
            frames = np.stack([resize(ds[t]) for t in seg])
            masked = br.apply(frames)
            W = TARGET[0]
            vw = cv2.VideoWriter(str(OUT / f"clip_{cam}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 12, (W * 2, TARGET[1]))
            for a, b in zip(frames, masked):
                vw.write(cv2.cvtColor(np.concatenate([a, b], 1), cv2.COLOR_RGB2BGR))
            vw.release()
            print("wrote", f"clip_{cam}.mp4")


if __name__ == "__main__":
    main()
