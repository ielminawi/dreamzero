"""
Compare background-removal backends on arm/hand/floor-heavy frames so we can verify
(black bg, keep table+arms+hands, remove red-tile floor) before full preprocessing.
Output: output/bg_compare/contact.png  (rows=frames; cols = original | segformer-b4 | mask2former)
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import cv2, h5py, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bg_removal  # noqa: E402

RAW = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries"
OUT = Path("/cluster/scratch/rjiang/dreamzero/output/bg_compare"); OUT.mkdir(parents=True, exist_ok=True)
TARGET = (640, 480)


def grab():
    files = sorted(glob.glob(f"{RAW}/*.h5"))
    picks = [(files[0], "aria_rgb_cam", 0.45), (files[0], "oakd_front_view", 0.45),
             (files[len(files)//3], "aria_rgb_cam", 0.5), (files[len(files)//3], "oakd_front_view", 0.5),
             (files[-1], "oakd_front_view", 0.55)]
    frames = []
    for fp, cam, frac in picks:
        with h5py.File(fp, "r") as f:
            ds = f[f"observations/images/{cam}/color"]; T = ds.shape[0]
            frames.append(cv2.resize(ds[int(T*frac)], TARGET, interpolation=cv2.INTER_AREA))
    return np.stack(frames)


def main():
    frames = grab()
    cols = [("original", None, None)]
    variants = [("segformer", bg_removal.DEFAULT_SEGFORMER), ("mask2former", bg_removal.DEFAULT_MASK2FORMER)]
    outs = {"original": frames}
    for backend, model in variants:
        br = bg_removal.BackgroundRemover(backend=backend, model_name=model, fill=0)
        print("running", backend, model, "device", br.device, flush=True)
        outs_b = br.apply(frames)
        outs[backend] = outs_b
        cols.append((backend, backend, model))
        del br
        import torch; torch.cuda.empty_cache()
    order = ["original", "segformer", "mask2former"]
    rows = []
    for i in range(len(frames)):
        rows.append(np.concatenate([cv2.resize(outs[k][i], (320, 240)) for k in order], axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(OUT / "contact.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("wrote", OUT / "contact.png", sheet.shape, "| cols: original | segformer-b4 | mask2former", flush=True)


if __name__ == "__main__":
    main()
