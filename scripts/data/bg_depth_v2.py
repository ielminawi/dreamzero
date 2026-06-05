"""
STANDALONE side-test v2: OPTIMIZED depth-based background removal (Depth Anything V2).

Still fully self-contained -- imports nothing from bg_removal.py / converter / eval,
so it cannot affect any running job. Writes contact sheets (and optional MP4) to output/.

What v2 improves over bg_depth_test.py (v1 = Otsu split + blind dilate+gaussian):
  1. MODEL: default Base (sharper depth edges than Small) -- override with --model.
  2. THRESHOLD: hybrid CLAMPED Otsu. Otsu finds the near/far split, but the kept
     fraction is clamped to [keep-min, keep-max] so it never over-cuts (eats the
     table / a far object) nor under-cuts (leaves a floor halo). Robust per-frame.
  3. EDGES: guided-filter matting. A small protective dilate guards thin arm tips,
     then a guided filter (RGB as guide) snaps the soft alpha to true object edges
     instead of a crude gaussian halo -> tight, clean boundaries, dark hands kept.
  4. (optional) TEMPORAL: --video applies an EMA on the threshold + alpha across a
     contiguous window to kill frame-to-frame flicker; writes a before/after MP4.

Depth Anything V2 outputs affine-invariant inverse depth: LARGER == CLOSER.

Run a quick quality comparison (v1 vs v2) on diverse frames, both cameras:
    python scripts/data/bg_depth_v2.py --num-frames 6 --cameras aria,oakd --model base
Temporal demo MP4 (contiguous window):
    python scripts/data/bg_depth_v2.py --video --num-frames 64 --cameras oakd --model base
"""
from __future__ import annotations

import argparse
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
FILL = 128
TARGET_WH = (640, 480)


# --------------------------- depth model ---------------------------
class DepthAnything:
    def __init__(self, model_key: str = "base", device: str | None = None):
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

    def depth(self, frames: np.ndarray, batch: int = 8) -> np.ndarray:
        torch = self.torch
        N, H, W, _ = frames.shape
        outs = []
        for s in range(0, N, batch):
            chunk = frames[s : s + batch]
            inp = self.proc(images=list(chunk), return_tensors="pt")
            inp = {k: (v.half() if (self.device == "cuda" and torch.is_floating_point(v)) else v).to(self.device)
                   for k, v in inp.items()}
            with torch.no_grad():
                pred = self.model(**inp).predicted_depth
            if pred.ndim == 4:
                pred = pred[:, 0]
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False
            )[:, 0]
            outs.append(pred.to("cpu").numpy().astype(np.float32))
        return np.concatenate(outs, 0)


# --------------------------- core ops ---------------------------
def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


def normalize_depth(d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    return np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def hybrid_threshold(dn: np.ndarray, keep_min: float, keep_max: float) -> float:
    """Otsu split on normalized depth, with kept (near) fraction clamped to
    [keep_min, keep_max]. Returns a threshold t in [0,1]; foreground = dn >= t."""
    u8 = (dn * 255).astype(np.uint8)
    t8, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = t8 / 255.0
    keep = float((dn >= t).mean())
    if keep < keep_min:                       # over-cut -> keep more (lower t)
        t = float(np.quantile(dn, 1.0 - keep_min))
    elif keep > keep_max:                     # under-cut -> keep less (raise t)
        t = float(np.quantile(dn, 1.0 - keep_max))
    return t


def clean_binary(fg: np.ndarray, close_radius: int = 6, min_island_frac: float = 0.004,
                 protect_dilate: int = 2) -> np.ndarray:
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
    if protect_dilate > 0:                    # guard thin arm tips before matting
        fg = ndi.binary_dilation(fg, structure=_disk(protect_dilate))
    return fg


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """Edge-aware refinement: pull soft alpha `src` onto `guide` (grayscale RGB) edges.
    Self-contained guided filter (He et al.) via box filters -- no ximgproc needed."""
    g = guide.astype(np.float32)
    p = src.astype(np.float32)
    k = (2 * radius + 1, 2 * radius + 1)
    box = lambda x: cv2.boxFilter(x, -1, k, normalize=True)
    mean_g, mean_p = box(g), box(p)
    cov_gp = box(g * p) - mean_g * mean_p
    var_g = box(g * g) - mean_g * mean_g
    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g
    return np.clip(box(a) * g + box(b), 0.0, 1.0)


def alpha_v2(frame: np.ndarray, dn: np.ndarray, keep_min: float, keep_max: float,
             gf_radius: int = 8, gf_eps: float = 1e-3) -> np.ndarray:
    t = hybrid_threshold(dn, keep_min, keep_max)
    fg = clean_binary(dn >= t)
    guide = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return guided_filter(guide, fg.astype(np.float32), gf_radius, gf_eps)


def fit_ground_plane(disp: np.ndarray, iters: int = 6, ds: int = 4) -> np.ndarray:
    """Robust (IRLS) fit of the dominant plane in disparity. A planar surface
    projects to disparity that is exactly affine in pixel coords, so the floor/
    table (largest smooth region) is captured; objects deviate. Returns the
    full-res plane prediction. Operates on a downsampled grid for speed."""
    h, w = disp.shape
    small = disp[::ds, ::ds].astype(np.float32)
    sh, sw = small.shape
    ys, xs = np.mgrid[0:sh, 0:sw]
    X = np.stack([xs.ravel() / sw, ys.ravel() / sh, np.ones(sh * sw, np.float32)], 1).astype(np.float32)
    d = small.ravel()
    wts = np.ones_like(d)
    beta = np.zeros(3, np.float32)
    for _ in range(iters):
        Wd = wts[:, None]
        beta, *_ = np.linalg.lstsq(X * Wd, d * wts, rcond=None)
        r = d - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        wts = 1.0 / (1.0 + (r / (2.0 * s)) ** 2)   # Cauchy weights -> robust to objects
    yf, xf = np.mgrid[0:h, 0:w]
    Xf = np.stack([xf.ravel() / w * (sw / sw), yf.ravel() / h, np.ones(h * w, np.float32)], 1)
    Xf[:, 0] = xf.ravel() / w
    return (Xf @ beta).reshape(h, w).astype(np.float32)


def alpha_plane(frame: np.ndarray, dn: np.ndarray, k: float = 1.0,
                gf_radius: int = 8, gf_eps: float = 1e-3) -> np.ndarray:
    """Ground-plane removal: keep pixels that rise ABOVE the fitted floor/table
    plane (closer than the plane -> objects/arms), drop pixels on/below it (floor,
    far wall). Distance-agnostic, so a far object on the table still survives."""
    plane = fit_ground_plane(dn)
    resid = dn - plane                      # >0 == closer than local plane == sticks up
    s = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
    fg = clean_binary(resid > k * s)
    guide = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return guided_filter(guide, fg.astype(np.float32), gf_radius, gf_eps)


def alpha_v1(frame: np.ndarray, dn: np.ndarray) -> np.ndarray:
    """Reproduce v1 (plain Otsu + blind dilate + gaussian feather) for comparison."""
    from scipy import ndimage as ndi

    u8 = (dn * 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = ndi.binary_fill_holes(u8 >= t)
    fg = ndi.binary_dilation(ndi.binary_closing(fg, structure=_disk(6)), structure=_disk(4))
    a = fg.astype(np.float32)
    return np.clip(cv2.GaussianBlur(a, (5, 5), 1.0), 0, 1)


def composite(frame: np.ndarray, alpha: np.ndarray, fill: int = FILL) -> np.ndarray:
    a = alpha[..., None]
    return (frame.astype(np.float32) * a + fill * (1.0 - a)).astype(np.uint8)


def depth_colormap(d: np.ndarray) -> np.ndarray:
    cm = cv2.applyColorMap((normalize_depth(d) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)


def label_row(imgs, titles) -> np.ndarray:
    out = []
    for im, t in zip(imgs, titles):
        im = im.copy()
        cv2.rectangle(im, (0, 0), (im.shape[1] - 1, 22), (0, 0, 0), -1)
        cv2.putText(im, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(im)
    return np.concatenate(out, axis=1)


def load_frames(h5_path: Path, key: str, n: int, contiguous: bool) -> np.ndarray:
    with h5py.File(h5_path, "r") as h:
        ds = h[key]
        T = ds.shape[0]
        if contiguous:
            start = int(0.4 * T)
            idx = np.arange(start, min(start + n, T))
        else:
            idx = np.linspace(int(0.1 * T), int(0.9 * T) - 1, n).astype(int)
        frames = np.stack([ds[i] for i in idx])
    return np.stack([cv2.resize(f, TARGET_WH) for f in frames])


# --------------------------- entry ---------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=RAW_DEFAULT)
    ap.add_argument("--h5", default=None)
    ap.add_argument("--cameras", default="aria,oakd")
    ap.add_argument("--num-frames", type=int, default=6)
    ap.add_argument("--model", choices=list(MODEL_IDS), default="base")
    ap.add_argument("--keep-min", type=float, default=0.30, help="min near fraction kept (anti over-cut)")
    ap.add_argument("--keep-max", type=float, default=0.70, help="max near fraction kept (anti floor-halo)")
    ap.add_argument("--gf-radius", type=int, default=8)
    ap.add_argument("--gf-eps", type=float, default=1e-3)
    ap.add_argument("--fill", type=int, default=FILL)
    ap.add_argument("--sweep", default=None,
                    help="comma keep-min values, e.g. '0.35,0.5,0.65'; renders one result column each "
                         "(conservativeness sweep) instead of the v1-vs-v2 comparison")
    ap.add_argument("--plane", action="store_true",
                    help="render orig|depth|hybrid|ground-plane-removal (distance-agnostic) comparison")
    ap.add_argument("--plane-k", type=float, default=1.0, help="ground-plane residual threshold (in robust stds)")
    ap.add_argument("--video", action="store_true", help="also write before/after MP4 w/ temporal EMA")
    ap.add_argument("--ema", type=float, default=0.6, help="temporal EMA weight on prev alpha (video)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    h5_path = Path(args.h5) if args.h5 else sorted(input_dir.glob("*.h5"))[0]
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    out_dir = Path(args.output) if args.output else Path("output/bg_depth_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[depth] h5={h5_path.name} cams={cams} model={args.model} "
          f"keep=[{args.keep_min},{args.keep_max}] video={args.video}")

    da = DepthAnything(args.model)

    for cam in cams:
        frames = load_frames(h5_path, CAMERA_KEYS[cam], args.num_frames, contiguous=args.video)
        depths = da.depth(frames)

        if args.video:
            prev = None
            vw = None
            vp = out_dir / f"video_{cam}_{args.model}.mp4"
            for f, d in zip(frames, depths):
                dn = normalize_depth(d)
                a = alpha_v2(f, dn, args.keep_min, args.keep_max, args.gf_radius, args.gf_eps)
                if prev is not None:
                    a = args.ema * prev + (1 - args.ema) * a
                prev = a
                res = composite(f, a, args.fill)
                pane = np.concatenate([f, res], axis=1)  # before | after
                if vw is None:
                    vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 20,
                                         (pane.shape[1], pane.shape[0]))
                vw.write(cv2.cvtColor(pane, cv2.COLOR_RGB2BGR))
            vw.release()
            print(f"[depth] wrote {vp}")
            continue

        sweep = [float(x) for x in args.sweep.split(",")] if args.sweep else None
        rows = []
        for f, d in zip(frames, depths):
            dn = normalize_depth(d)
            if sweep:                              # conservativeness sweep: higher keep-min => keeps more
                imgs = [f, depth_colormap(d)]
                titles = [f"{cam} orig", f"depth ({args.model})"]
                for km in sweep:
                    a = alpha_v2(f, dn, km, max(args.keep_max, km + 0.05), args.gf_radius, args.gf_eps)
                    imgs.append(composite(f, a, args.fill))
                    titles.append(f"keep-min {km:.2f}")
                rows.append(label_row(imgs, titles))
            elif args.plane:                       # hybrid threshold vs ground-plane removal
                res_h = composite(f, alpha_v2(f, dn, args.keep_min, args.keep_max,
                                              args.gf_radius, args.gf_eps), args.fill)
                res_p = composite(f, alpha_plane(f, dn, args.plane_k, args.gf_radius, args.gf_eps), args.fill)
                rows.append(label_row(
                    [f, depth_colormap(d), res_h, res_p],
                    [f"{cam} orig", f"depth ({args.model})", "hybrid threshold", f"ground-plane k={args.plane_k}"],
                ))
            else:
                res_v1 = composite(f, alpha_v1(f, dn), args.fill)
                res_v2 = composite(f, alpha_v2(f, dn, args.keep_min, args.keep_max,
                                               args.gf_radius, args.gf_eps), args.fill)
                rows.append(label_row(
                    [f, depth_colormap(d), res_v1, res_v2],
                    [f"{cam} orig", f"depth ({args.model})", "v1 otsu+dilate", "v2 hybrid+guided"],
                ))
        sheet = np.concatenate(rows, axis=0)
        p = out_dir / f"contact_{cam}_{args.model}.png"
        cv2.imwrite(str(p), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(f"[depth] wrote {p}  ({sheet.shape[1]}x{sheet.shape[0]})")

    print("[depth] done.")


if __name__ == "__main__":
    main()
