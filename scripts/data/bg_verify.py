"""
Verify Mask2Former background-removal mask COVERAGE (esp. the dark Orca arms/hands).
Renders removed-bg as bright green so kept-vs-removed is unambiguous even for dark arms.
Output: output/bg_verify/contact.png  (rows=frames; cols = original | green-overlay | black-fill)
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import cv2, h5py, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bg_removal  # noqa: E402

RAW = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries"
OUT = Path("/cluster/scratch/rjiang/dreamzero/output/bg_verify"); OUT.mkdir(parents=True, exist_ok=True)
TARGET = (640, 480)
GREEN = np.array([0, 255, 0], np.float32)


def grab():
    files = sorted(glob.glob(f"{RAW}/*.h5"))
    # emphasise the aria (ego) view with dark arms, a few times + episodes
    picks = []
    for fp in [files[0], files[len(files)//2]]:
        for cam, fracs in [("aria_rgb_cam", [0.25, 0.45, 0.7]), ("oakd_front_view", [0.45])]:
            for fr in fracs:
                picks.append((fp, cam, fr))
    out = []
    for fp, cam, fr in picks:
        with h5py.File(fp, "r") as f:
            ds = f[f"observations/images/{cam}/color"]; T = ds.shape[0]
            out.append(cv2.resize(ds[int(T*fr)], TARGET, interpolation=cv2.INTER_AREA))
    return np.stack(out)


def main():
    frames = grab()
    br = bg_removal.BackgroundRemover(backend="mask2former", fill=0)
    print("device", br.device, "model", br.model_name, "remove_n", len(br.remove_ids), flush=True)
    alpha = br.foreground_alpha(frames)[..., None]
    rows = []
    for i in range(len(frames)):
        a = alpha[i]
        green = (frames[i] * a + GREEN * (1 - a)).astype(np.uint8)   # removed -> green
        black = (frames[i] * a).astype(np.uint8)                     # removed -> black (final)
        trio = [cv2.resize(x, (320, 240)) for x in (frames[i], green, black)]
        rows.append(np.concatenate(trio, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(OUT / "contact.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("wrote", OUT / "contact.png", sheet.shape, "| cols: original | green-overlay(removed) | black-fill", flush=True)


if __name__ == "__main__":
    main()
