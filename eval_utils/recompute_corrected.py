#!/usr/bin/env python3
"""Recompute WM-vs-GT metrics with the CORRECT 2x2 view layout, reusing the already
-decoded predictions baked into the *_aria_rgb_cam_GTvsPRED.mp4 side-by-side videos
(whose PRED half == the composite LEFT column = aria(top)+oakd(bottom)). No GPU.

Note: predictions here come from H.264-compressed mp4s, so metrics are approximate
(a clean GPU re-run with the fixed split gives exact numbers); the point is to
confirm both views are predicted and produce a corrected filmstrip immediately.
"""
import argparse, glob, json, os
import cv2, imageio.v2 as iio, numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

DCAM = {"aria_rgb_cam": "observation.images.aria_rgb_cam",
        "oakd_front_view": "observation.images.oakd_front_view"}


def gt_window(dsdir, ep, cam, start, n):
    import decord
    p = f"{dsdir}/videos/chunk-000/{DCAM[cam]}/episode_{ep:06d}.mp4"
    vr = decord.VideoReader(p)
    idx = list(range(start, min(start + n, len(vr))))
    fr = vr.get_batch(idx).asnumpy()
    if len(fr) < n:
        fr = np.concatenate([fr, np.repeat(fr[-1:], n - len(fr), 0)], 0)
    return fr


def rs(img, h, w):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def lab(img, t, col=(0, 255, 0)):
    img = img.copy(); cv2.rectangle(img, (0, 0), (img.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(img, t, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--film-clip", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    rows = []
    summ = []
    for mj in sorted(glob.glob(f"{a.in_dir}/*_metrics.json")):
        m = json.load(open(mj))
        ep, start, off = m["episode"], m["start"], m["align_offset"]
        clip = f"ep{ep:06d}_s{start:05d}"
        sbs = f"{a.in_dir}/{clip}_aria_rgb_cam_GTvsPRED.mp4"
        if not os.path.isfile(sbs):
            continue
        r = iio.get_reader(sbs); fr = np.stack([f for f in r]); r.close()
        leftcol = fr[:, :, fr.shape[2] // 2:]          # composite cols 0:320 (aria top, oakd bottom)
        H = leftcol.shape[1]; hh = H // 2
        pred = {"aria_rgb_cam": leftcol[:, :hh], "oakd_front_view": leftcol[:, hh:hh * 2]}
        vh, vw = hh, leftcol.shape[2]
        n = pred["aria_rgb_cam"].shape[0]
        rec = {"episode": ep, "start": start}
        per = {}
        for cam in ["aria_rgb_cam", "oakd_front_view"]:
            gt = gt_window(a.dataset_dir, ep, cam, start, n + off)
            gt = np.stack([rs(f, vh, vw) for f in gt])[off:off + n]
            pr = pred[cam][:gt.shape[0]]; gt = gt[:pr.shape[0]]
            ps = [float(psnr(gt[i], pr[i], data_range=255)) for i in range(len(gt))]
            ss = [float(ssim(gt[i], pr[i], data_range=255, channel_axis=2)) for i in range(len(gt))]
            per[cam] = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss))}
            rec[f"{cam}_psnr"] = per[cam]["psnr"]; rec[f"{cam}_ssim"] = per[cam]["ssim"]
            sb = np.concatenate([gt, pr], axis=2)
            iio.mimsave(f"{a.out_dir}/{clip}_{cam}_GTvsPRED.mp4", list(sb), fps=10, codec="libx264")
        summ.append(rec)
        print(f"{clip}: aria PSNR={per['aria_rgb_cam']['psnr']:.2f} SSIM={per['aria_rgb_cam']['ssim']:.3f} | "
              f"oakd PSNR={per['oakd_front_view']['psnr']:.2f} SSIM={per['oakd_front_view']['ssim']:.3f}")

    # aggregate
    import csv
    with open(f"{a.out_dir}/summary_corrected.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode", "start", "aria_psnr", "aria_ssim", "oakd_psnr", "oakd_ssim"])
        for r in summ:
            w.writerow([r["episode"], r["start"], f"{r['aria_rgb_cam_psnr']:.3f}", f"{r['aria_rgb_cam_ssim']:.4f}",
                        f"{r['oakd_front_view_psnr']:.3f}", f"{r['oakd_front_view_ssim']:.4f}"])
    for cam in ["aria_rgb_cam", "oakd_front_view"]:
        print(f"OVERALL {cam}: PSNR={np.mean([r[cam+'_psnr'] for r in summ]):.2f} "
              f"SSIM={np.mean([r[cam+'_ssim'] for r in summ]):.3f}")

    # filmstrip for one clip: 4 rows (aria GT/PRED, oakd GT/PRED)
    clip = a.film_clip or f"ep{summ[0]['episode']:06d}_s{summ[0]['start']:05d}"
    cols = 8
    strips = []
    for cam in ["aria_rgb_cam", "oakd_front_view"]:
        r = iio.get_reader(f"{a.out_dir}/{clip}_{cam}_GTvsPRED.mp4"); fr = np.stack([x for x in r]); r.close()
        half = fr.shape[2] // 2; gt = fr[:, :, :half]; pr = fr[:, :, half:half * 2]
        idx = np.linspace(0, len(fr) - 1, cols).astype(int)
        pad = 3
        def mkrow(stack, name, col):
            cells = []
            for k, i in enumerate(idx):
                cells.append(lab(stack[i], f"{name} t={i}", col))
                if k < cols - 1: cells.append(np.zeros((stack.shape[1], pad, 3), np.uint8))
            return np.concatenate(cells, 1)
        strips.append(mkrow(gt, f"{cam[:4]} GT", (0, 255, 0)))
        strips.append(np.zeros((pad, strips[-1].shape[1], 3), np.uint8))
        strips.append(mkrow(pr, f"{cam[:4]} PRED", (0, 200, 255)))
        strips.append(np.zeros((8, strips[-1].shape[1], 3), np.uint8))
    W = max(s.shape[1] for s in strips)
    strips = [np.pad(s, ((0, 0), (0, W - s.shape[1]), (0, 0))) for s in strips]
    banner = np.zeros((30, W, 3), np.uint8)
    cv2.putText(banner, f"DreamZero WM ckpt-20000  {clip}  | CORRECTED 2x2 split: BOTH views predicted",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    fig = np.concatenate([banner] + strips, 0)
    iio.imwrite(f"{a.out_dir}/filmstrip_corrected_{clip}.png", fig)
    print("wrote filmstrip", f"{a.out_dir}/filmstrip_corrected_{clip}.png", fig.shape)


if __name__ == "__main__":
    main()
