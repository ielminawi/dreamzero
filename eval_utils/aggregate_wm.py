#!/usr/bin/env python3
"""Aggregate validate_wm.py per-clip metrics into a summary CSV, horizon-degradation
curve (PNG), and a markdown report."""
import argparse
import csv
import json
import os

import numpy as np


def load_clips(out_dir):
    p = os.path.join(out_dir, "all_clips.json")
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    clips = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith("_metrics.json"):
            with open(os.path.join(out_dir, fn)) as f:
                clips.append(json.load(f))
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default="checkpoint-20000")
    args = ap.parse_args()

    clips = load_clips(args.out_dir)
    if not clips:
        print("No clips found.")
        return
    cams = list(clips[0]["cameras"].keys())

    # ── per-clip summary CSV ──
    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["episode", "start", "align_offset", "n_eval"]
        for c in cams:
            head += [f"{c}_psnr", f"{c}_ssim", f"{c}_mse"]
        w.writerow(head)
        for cl in clips:
            row = [cl["episode"], cl["start"], cl["align_offset"], cl["n_eval"]]
            for c in cams:
                cc = cl["cameras"][c]
                row += [f"{cc['psnr_mean']:.3f}", f"{cc['ssim_mean']:.4f}", f"{cc['mse_mean']:.2f}"]
            w.writerow(row)

    # ── overall + per-horizon aggregation ──
    overall = {c: {"psnr": [], "ssim": [], "mse": []} for c in cams}
    maxh = max(cl["n_eval"] for cl in clips)
    horizon = {c: {"psnr": [[] for _ in range(maxh)], "ssim": [[] for _ in range(maxh)]} for c in cams}
    for cl in clips:
        for c in cams:
            cc = cl["cameras"][c]
            overall[c]["psnr"].append(cc["psnr_mean"])
            overall[c]["ssim"].append(cc["ssim_mean"])
            overall[c]["mse"].append(cc["mse_mean"])
            for i, (ps, ss) in enumerate(zip(cc["psnr"], cc["ssim"])):
                horizon[c]["psnr"][i].append(ps)
                horizon[c]["ssim"][i].append(ss)

    # ── degradation curve PNG ──
    png_path = os.path.join(args.out_dir, "horizon_curve.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for c in cams:
            ph = [np.mean(x) if x else np.nan for x in horizon[c]["psnr"]]
            sh = [np.mean(x) if x else np.nan for x in horizon[c]["ssim"]]
            axes[0].plot(range(len(ph)), ph, marker="o", label=c)
            axes[1].plot(range(len(sh)), sh, marker="o", label=c)
        axes[0].set_title("PSNR vs prediction-horizon frame"); axes[0].set_xlabel("pred frame"); axes[0].set_ylabel("PSNR (dB)"); axes[0].grid(True, alpha=.3); axes[0].legend()
        axes[1].set_title("SSIM vs prediction-horizon frame"); axes[1].set_xlabel("pred frame"); axes[1].set_ylabel("SSIM"); axes[1].grid(True, alpha=.3); axes[1].legend()
        fig.suptitle(f"World-model video prediction vs GT — {args.checkpoint}")
        fig.tight_layout()
        fig.savefig(png_path, dpi=110)
    except Exception as e:
        png_path = f"(plot skipped: {e})"

    # ── markdown report ──
    md = os.path.join(args.out_dir, "REPORT.md")
    with open(md, "w") as f:
        f.write(f"# World-Model Video-Prediction Validation — {args.checkpoint}\n\n")
        f.write(f"Clips evaluated: **{len(clips)}**  |  cameras: {', '.join(cams)}\n\n")
        f.write("Protocol: action-free, video-conditioned autoregressive prediction "
                "(`GrootSimPolicy.lazy_joint_forward_causal`), the same path that drives "
                "the eval server. Predicted frames are temporally aligned to ground truth "
                "(SSIM-max offset) and compared per camera view at the model's native "
                "decode resolution.\n\n")
        f.write("## Overall (mean over clips)\n\n")
        f.write("| Camera | PSNR (dB) | SSIM | MSE |\n|---|---|---|---|\n")
        for c in cams:
            f.write(f"| {c} | {np.mean(overall[c]['psnr']):.2f} | "
                    f"{np.mean(overall[c]['ssim']):.3f} | {np.mean(overall[c]['mse']):.1f} |\n")
        f.write("\n## Per-clip\n\nSee `summary.csv`. Side-by-side `[GT | PRED]` videos: "
                "`*_GTvsPRED.mp4`.\n\n")
        f.write(f"## Horizon degradation\n\n![horizon]({os.path.basename(png_path)})\n\n")
        f.write("PSNR/SSIM as a function of how far into the future each predicted "
                "frame is — the key signal for world-model fidelity (higher = closer "
                "to reality; expected to decay with horizon).\n")

    print(f"Wrote:\n  {csv_path}\n  {png_path}\n  {md}")
    for c in cams:
        print(f"  {c}: PSNR={np.mean(overall[c]['psnr']):.2f}dB "
              f"SSIM={np.mean(overall[c]['ssim']):.3f} MSE={np.mean(overall[c]['mse']):.1f}")


if __name__ == "__main__":
    main()
