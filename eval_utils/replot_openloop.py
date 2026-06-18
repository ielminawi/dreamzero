#!/usr/bin/env python3
"""Re-render a *presentable* open-loop comparison figure from already-saved rollout
mp4s (no GPU / model load needed).

Rebuilds results[mode][cam] by reading the per-mode rollout videos written by
openloop.py, reads GT straight from the dataset, then draws a cleaner grid:

    rows    : Ground truth | Sim (GT actions) | Open loop | Closed loop
    columns : t = 5,10,15,20,25
"""
import argparse, os, sys
import numpy as np
import imageio.v2 as iio

sys.path.insert(0, os.path.dirname(__file__))
from validate_wm import CAMERA_KEYS, CONCAT_ORDER, read_gt_window, resize_to

# mode -> (bold label, gray sub-descriptor)
ROW_LABELS = {
    "gtsim":  ("Sim (GT actions)", "world model driven by recorded actions"),
    "dream":  ("Open loop",        "model dreams its own actions (fused)"),
    "polact": ("Closed loop",      "policy → action → world model"),
}


def read_quads(path):
    """rollout mp4 -> (T,H,W,3) uint8 (each frame already a single-camera quad)."""
    rd = iio.get_reader(path)
    frames = np.stack([f for f in rd])
    rd.close()
    return frames


def best_off(gt, pr, maxoff=6):
    from skimage.metrics import structural_similarity as ssim
    P, G = pr.shape[0], gt.shape[0]
    bo, bs = 0, -1e9
    for o in range(maxoff + 1):
        n = min(P, G - o)
        if n <= 6:
            continue
        s = np.mean([ssim(gt[o + k], pr[k], data_range=255, channel_axis=2)
                     for k in range(0, min(n, 12))])
        if s > bs:
            bs, bo = s, o
    return bo


def ssim_color(v, lo=0.30, hi=0.85):
    """red (low) -> amber -> green (high) badge background."""
    import matplotlib.cm as cm
    t = float(np.clip((v - lo) / (hi - lo), 0, 1))
    r, g, b, _ = cm.RdYlGn(t)
    return (r, g, b)


def plot_one_cam(cam, results, gt_v, horizons, ckpt, ep, start, out_png, fps=50.0,
                 col_label="seconds", latex_font=False, col_desc=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.metrics import structural_similarity as ssim

    if latex_font:
        # Computer Modern (matplotlib's bundled cmr10) -- LaTeX look, no TeX needed
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,   # cmr10 has no unicode minus glyph
        })

    # cmr10 lacks the · and → glyphs -- render them via mathtext instead
    tx = ((lambda s: s.replace("·", r"$\cdot$").replace("→", r"$\to$"))
          if latex_font else (lambda s: s))

    modes = list(results.keys())
    rows = 1 + len(modes)
    ncol = len(horizons)

    offs = {m: best_off(gt_v, results[m][cam]) for m in modes}

    cell = 1.9
    fig, ax = plt.subplots(rows, ncol,
                           figsize=(cell * ncol + 1.6, cell * rows + 0.9),
                           squeeze=False)
    fig.patch.set_facecolor("white")

    def style(a):
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor("#cfcfcf"); s.set_linewidth(0.8)

    # ---- ground-truth row ----
    for ci, t in enumerate(horizons):
        a = ax[0][ci]
        a.imshow(gt_v[min(t, gt_v.shape[0] - 1)]); style(a)
        title = f"Frame {t}" if col_label == "frames" else f"{t / fps:g} s"
        a.set_title(title, fontsize=14,
                    fontweight="medium", pad=6, color="#222222")
    ax[0][0].annotate("Ground truth", xy=(-0.085, 0.5), xycoords="axes fraction",
                      rotation=90, ha="center", va="center",
                      fontsize=13, fontweight="bold", color="#111111")

    # ---- prediction rows ----
    for mi, m in enumerate(modes):
        q = results[m][cam]; off = offs[m]; T = q.shape[0]
        bold, sub = ROW_LABELS.get(m, (m, ""))
        for ci, t in enumerate(horizons):
            a = ax[mi + 1][ci]
            pt = min(t, T - 1)
            prf = q[pt]
            gtf = gt_v[min(off + t, gt_v.shape[0] - 1)]
            sv = ssim(gtf, prf, data_range=255, channel_axis=2)
            a.imshow(prf); style(a)
            a.text(0.97, 0.06, f"SSIM {sv:.2f}", transform=a.transAxes,
                   fontsize=8.5, ha="right", va="bottom", color="#111111",
                   bbox=dict(boxstyle="round,pad=0.22", fc=ssim_color(sv),
                             ec="none", alpha=0.92))
        ax[mi + 1][0].annotate(bold, xy=(-0.085, 0.5), xycoords="axes fraction",
                               rotation=90, ha="center", va="center",
                               fontsize=13, fontweight="bold", color="#111111")

    pretty_cam = {"aria_rgb_cam": "Aria RGB (head)",
                  "oakd_front_view": "OAK-D (front)"}.get(cam, cam)
    fig.suptitle(f"Open-loop world-model rollout vs ground truth",
                 fontsize=16, fontweight="bold", y=0.995, color="#111111")
    sub = f"{pretty_cam}   ·   {ckpt}   ·   episode {ep}, frame {start}"
    if col_desc:
        sub += ("   ·   columns = rollout frame index" if col_label == "frames"
                else f"   ·   columns = open-loop rollout time (seconds @ {fps:g} fps)")
    fig.text(0.5, 0.952, tx(sub),
             ha="center", fontsize=10.5, color="#666666")

    fig.subplots_adjust(left=0.075, right=0.992, top=0.915, bottom=0.058,
                        wspace=0.04, hspace=0.06)

    # divider between GT row and predictions
    p0 = ax[0][0].get_position(); p1 = ax[1][0].get_position()
    yline = (p0.y0 + p1.y1) / 2
    fig.lines.append(plt.Line2D([0.04, 0.995], [yline, yline],
                                transform=fig.transFigure,
                                color="#bbbbbb", lw=1.0, ls=(0, (4, 3))))

    # footnote legend (row meanings, one clean line instead of cramped vertical text)
    legend = "   ·   ".join(f"{ROW_LABELS[m][0]}: {ROW_LABELS[m][1]}"
                            for m in modes if m in ROW_LABELS)
    fig.text(0.5, 0.022, tx(legend), ha="center", va="center",
             fontsize=8.5, color="#777777")
    fig.text(0.5, 0.005,
             tx("SSIM badge = similarity to time-aligned ground truth (red low → green high)"),
             ha="center", va="center", fontsize=8, color="#999999")

    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll-dir", required=True,
                    help="dir containing the saved rollout mp4s")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--clip", default="32:1600",
                    help="ep:start, or comma-separated list 'ep:start,ep:start,...'")
    ap.add_argument("--modes", default="gtsim,dream,polact")
    ap.add_argument("--horizons", default="5,10,15,20,25")
    ap.add_argument("--ckpt", default="checkpoint-20000")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cams", default=None,
                    help="comma-separated camera filter (default: all)")
    ap.add_argument("--col-label", choices=["seconds", "frames"], default="seconds")
    ap.add_argument("--out-suffix", default="",
                    help="appended to the output filename stem")
    ap.add_argument("--latex-font", action="store_true",
                    help="render text in Computer Modern (LaTeX paper style)")
    ap.add_argument("--no-col-desc", action="store_true",
                    help="drop the 'columns = ...' note from the subtitle")
    a = ap.parse_args()

    clips = [tuple(int(x) for x in c.split(":")) for c in a.clip.split(",")]
    modes = a.modes.split(",")
    horizons = [int(x) for x in a.horizons.split(",")]
    out_dir = a.out_dir or a.roll_dir
    os.makedirs(out_dir, exist_ok=True)

    fps = 50.0
    try:
        import json
        fps = float(json.load(open(os.path.join(a.dataset_dir, "meta", "info.json")))["fps"])
    except Exception:
        pass

    for ep, start in clips:
        tag = f"ep{ep:06d}_s{start:05d}"
        # reconstruct results[mode][cam] from the saved rollout mp4s
        results, hw = {}, None
        for m in modes:
            results[m] = {}
            for cam in CONCAT_ORDER:
                p = os.path.join(a.roll_dir, f"{tag}_{m}_{cam}.mp4")
                results[m][cam] = read_quads(p)
                hw = results[m][cam].shape[1:3]
        hh, ww = hw

        Tmax = max(results[m][CONCAT_ORDER[0]].shape[0] for m in modes)
        gt = read_gt_window(a.dataset_dir, ep, CAMERA_KEYS, start,
                            Tmax + max(horizons) + 8)
        gt_v = {cam: np.stack([resize_to(f, hh, ww) for f in gt[ck]])
                for ck, cam in zip(CAMERA_KEYS, CONCAT_ORDER)}

        cams = a.cams.split(",") if a.cams else CONCAT_ORDER
        for cam in cams:
            out_png = os.path.join(out_dir,
                                   f"openloop_pretty_{tag}_{cam}{a.out_suffix}.png")
            plot_one_cam(cam, results, gt_v[cam], horizons, a.ckpt, ep, start,
                         out_png, fps=fps, col_label=a.col_label,
                         latex_font=a.latex_font, col_desc=not a.no_col_desc)


if __name__ == "__main__":
    main()
