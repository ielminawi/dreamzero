#!/usr/bin/env python3
"""Build a GT-vs-prediction filmstrip PNG from the *_GTvsPRED.mp4 side-by-side
videos produced by validate_wm.py. Each side-by-side frame is [GT | PRED] (width
split in half); we sample N timesteps and tile GT row over PRED row, per camera."""
import argparse
import os

import cv2
import imageio.v2 as iio
import numpy as np


def load_split(path):
    r = iio.get_reader(path)
    frames = [f for f in r]
    r.close()
    arr = np.stack(frames)  # (T, H, W, 3) ; W = 2*view_w
    half = arr.shape[2] // 2
    return arr[:, :, :half], arr[:, :, half:half * 2]  # gt, pred


def label(img, text, color=(255, 255, 255)):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def strip(gt, pred, n_cols, cam, off):
    T = gt.shape[0]
    idxs = np.linspace(0, T - 1, n_cols).astype(int)
    pad = 3
    cols = []
    for j, i in enumerate(idxs):
        g = label(gt[i], f"t={i}")
        p = pred[i]
        col = np.concatenate(
            [g, np.zeros((pad, g.shape[1], 3), np.uint8), p], axis=0
        )
        cols.append(col)
        if j < n_cols - 1:
            cols.append(np.zeros((col.shape[0], pad, 3), np.uint8))
    row = np.concatenate(cols, axis=1)
    # left margin with GT/PRED labels
    margin = np.zeros((row.shape[0], 70, 3), np.uint8)
    h = gt.shape[1]
    cv2.putText(margin, "GROUND", (4, h // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(margin, "TRUTH", (4, h // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(margin, "PRED", (4, h + 3 + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
    row = np.concatenate([margin, row], axis=1)
    title = np.zeros((26, row.shape[1], 3), np.uint8)
    cv2.putText(title, f"{cam}   (align offset={off})", (74, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([title, row], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--clip", required=True, help="e.g. ep000005_s00050")
    ap.add_argument("--cams", default="aria_rgb_cam,oakd_front_view")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subtitle", default="GT-conditioned autoregressive rollout")
    args = ap.parse_args()

    import json
    off = {}
    mp = os.path.join(args.dir, f"{args.clip}_metrics.json")
    if os.path.isfile(mp):
        off = json.load(open(mp)).get("align_offset", 0)

    rows = []
    for cam in args.cams.split(","):
        gt, pred = load_split(os.path.join(args.dir, f"{args.clip}_{cam}_GTvsPRED.mp4"))
        rows.append(strip(gt, pred, args.cols, cam, off))
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in rows]
    banner = np.zeros((30, W, 3), np.uint8)
    cv2.putText(banner, f"DreamZero WM  {args.clip}  |  {args.subtitle}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    fig = np.concatenate([banner] + [r for pair in zip(rows, [np.zeros((6, W, 3), np.uint8)] * len(rows)) for r in pair][:-1], axis=0)
    iio.imwrite(args.out, fig)
    print("wrote", args.out, fig.shape)


if __name__ == "__main__":
    main()
