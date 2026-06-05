"""
STANDALONE side-test: depth-based background removal via Depth Anything V2.

Purpose: *just look at how good it is* before wiring anything into the real
pipeline. This script is fully self-contained -- it does NOT import or modify
bg_removal.py, the converter, or the eval path, so it cannot affect any running
job. It only reads the raw H5 and writes a contact sheet under output/.

Idea (per supervisor): instead of brittle ADE20K semantic segmentation, estimate
per-frame monocular depth (Depth Anything V2, pure PyTorch -> fits the cu12.9 env)
and cut the FAR background by thresholding distance. The near-field manipulation
scene (table + arms + hands + objects) is kept; the room behind is replaced with a
constant gray fill (128, matching the existing pipeline so a dark Orca hand stays
visible).

Depth Anything V2 outputs affine-invariant inverse depth: LARGER value == CLOSER.
So foreground == the high-value (near) pixels. The split is per-frame adaptive
(Otsu by default; or keep-nearest fraction) because monocular depth has no fixed
metric scale across views/frames.

Run (CPU is fine for a few frames; Small model downloads from HF on first use):
    python scripts/data/bg_depth_test.py \
        --num-frames 6 --cameras aria,oakd --model small --mode otsu

Compare modes / model sizes:
    python scripts/data/bg_depth_test.py --model base --mode both --keep-frac 0.55
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

RAW_DEFAULT = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries"
CAMERA_KEYS = {
    "aria": "observations/images/aria_rgb_cam/color",
    "oakd": "observations/images/oakd_front_view/color",
}
MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}
FILL = 128            # neutral gray (same as production bg_removal.DEFAULT_FILL)
TARGET_WH = (640, 480)  # match the converter's --target-resolution 640x480


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


def clean_mask(fg: np.ndarray, close_radius: int = 6, dilate_radius: int = 4,
               min_island_frac: float = 0.004) -> np.ndarray:
    """Same spirit as bg_removal._clean_lowres: fill holes, close, drop speckle,
    DILATE so the full arm survives. Operates on the depth-derived foreground."""
    from scipy import ndimage as ndi

    h, w = fg.shape
    fg = ndi.binary_fill_holes(fg)
    if close_radius > 0:
        fg = ndi.binary_closing(fg, structure=_disk(close_radius))
        fg = ndi.binary_fill_holes(fg)
    lbl, n = ndi.label(fg)
    if n > 0:
        sizes = np.bincount(lbl.ravel())
        small = np.where(sizes < min_island_frac * h * w)[0]
        if small.size:
            fg &= ~np.isin(lbl, small)
    fg = ndi.binary_fill_holes(fg)
    if dilate_radius > 0:
        fg = ndi.binary_dilation(fg, structure=_disk(dilate_radius))
    return fg


class DepthAnything:
    def __init__(self, model_key: str = "small", device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        mid = MODEL_IDS[model_key]
        print(f"[depth] loading {mid} on {self.device} ...", flush=True)
        self.proc = AutoImageProcessor.from_pretrained(mid)
        self.model = AutoModelForDepthEstimation.from_pretrained(mid).eval().to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()

    def depth(self, frames: np.ndarray) -> np.ndarray:
        """frames (N,H,W,3) uint8 RGB -> (N,H,W) float32 inverse-depth (high==near)."""
        torch = self.torch
        N, H, W, _ = frames.shape
        inp = self.proc(images=list(frames), return_tensors="pt")
        inp = {k: (v.half() if (self.device == "cuda" and torch.is_floating_point(v)) else v).to(self.device)
               for k, v in inp.items()}
        with torch.no_grad():
            pred = self.model(**inp).predicted_depth  # (N,h,w) or (N,1,h,w)
        if pred.ndim == 4:
            pred = pred[:, 0]
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False
        )[:, 0]
        return pred.to("cpu").numpy().astype(np.float32)


def foreground_from_depth(depth: np.ndarray, mode: str, keep_frac: float) -> np.ndarray:
    """depth (H,W) high==near. Return boolean foreground (near) mask.
    mode='otsu' -> adaptive bimodal split; mode='frac' -> keep nearest keep_frac."""
    d = depth.astype(np.float32)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    if mode == "frac":
        thr = np.quantile(dn, 1.0 - keep_frac)
        return dn >= thr
    # otsu on 8-bit normalized depth; keep the NEAR (bright) side
    u8 = (dn * 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return u8 >= t


def apply_fill(frame: np.ndarray, fg: np.ndarray, feather: float = 1.0, fill: int = FILL) -> np.ndarray:
    alpha = fg.astype(np.float32)
    if feather > 0:
        k = max(1, int(feather * 4) | 1)
        alpha = cv2.GaussianBlur(alpha, (k, k), feather)
    alpha = np.clip(alpha, 0, 1)[..., None]
    out = frame.astype(np.float32) * alpha + fill * (1.0 - alpha)
    return out.astype(np.uint8)


def depth_colormap(depth: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(depth, 1), np.percentile(depth, 99)
    dn = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    cm = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)  # BGR
    return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)


def label_row(imgs: list[np.ndarray], titles: list[str]) -> np.ndarray:
    out = []
    for im, t in zip(imgs, titles):
        im = im.copy()
        cv2.rectangle(im, (0, 0), (im.shape[1] - 1, 22), (0, 0, 0), -1)
        cv2.putText(im, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(im)
    return np.concatenate(out, axis=1)


def sample_frames(h5_path: Path, key: str, n: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as h:
        ds = h[key]
        T = ds.shape[0]
        idx = np.linspace(int(0.1 * T), int(0.9 * T) - 1, n).astype(int)
        frames = np.stack([ds[i] for i in idx])  # (n,Hn,Wn,3)
    return np.stack([cv2.resize(f, TARGET_WH) for f in frames])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=RAW_DEFAULT)
    ap.add_argument("--h5", default=None, help="specific .h5; default = first in input-dir")
    ap.add_argument("--cameras", default="aria,oakd")
    ap.add_argument("--num-frames", type=int, default=6)
    ap.add_argument("--model", choices=list(MODEL_IDS), default="small")
    ap.add_argument("--mode", choices=["otsu", "frac", "both"], default="otsu")
    ap.add_argument("--keep-frac", type=float, default=0.55, help="for mode=frac/both: nearest fraction kept")
    ap.add_argument("--fill", type=int, default=FILL)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    h5_path = Path(args.h5) if args.h5 else sorted(input_dir.glob("*.h5"))[0]
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    out_dir = Path(args.output) if args.output else Path("output/bg_depth_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[depth] h5={h5_path.name} cams={cams} frames={args.num_frames} model={args.model} mode={args.mode}")

    da = DepthAnything(args.model)
    modes = ["otsu", "frac"] if args.mode == "both" else [args.mode]

    for cam in cams:
        frames = sample_frames(h5_path, CAMERA_KEYS[cam], args.num_frames)  # (n,480,640,3)
        depths = da.depth(frames)
        rows = []
        for f, d in zip(frames, depths):
            imgs = [f, depth_colormap(d)]
            titles = [f"{cam} orig", "depth (near=bright)"]
            for m in modes:
                fg = clean_mask(foreground_from_depth(d, m, args.keep_frac))
                res = apply_fill(f, fg, fill=args.fill)
                imgs.append(res)
                titles.append(f"bg-removed [{m}]" + (f" {args.keep_frac:.2f}" if m == "frac" else ""))
            rows.append(label_row(imgs, titles))
        sheet = np.concatenate(rows, axis=0)
        p = out_dir / f"contact_{cam}_{args.model}.png"
        cv2.imwrite(str(p), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(f"[depth] wrote {p}  ({sheet.shape[1]}x{sheet.shape[0]})")

    print("[depth] done. Open the contact sheets to judge quality.")


if __name__ == "__main__":
    main()
