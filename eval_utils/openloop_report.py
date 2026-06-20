#!/usr/bin/env python
"""Consolidated NON-SIM open-loop characterization of the varnorm policy across checkpoints.
For each checkpoint's action_eval output (action_accuracy.json + all_traces.npz) reports:
  - high-motion deploy verdict: reanchored MAE vs stay (offset-corrected)
  - MOTION CAPTURE: ||pred_disp|| / ||gt_disp|| on high-motion chunks  (>1 over-shoot, <1 under-predict)
  - DIRECTION: cosine(pred_disp, gt_disp) on high-motion chunks  (1=perfect heading, 0=random)
  - per-DIMENSION over/under-shoot ratio (which joints misbehave) from raw traces
Usage: python openloop_report.py output/action_eval_varnorm_1000_full [..._2000_full ...]
"""
import sys, json, os, numpy as np

SEGMENTS = [("left_arm_joint_pos", 0, 7), ("right_arm_joint_pos", 7, 14),
            ("left_hand_joint_pos", 14, 31), ("right_hand_joint_pos", 31, 48)]

def motion_stats(npz, aa, bb, hi_frac=0.33):
    """From raw chunk traces, compute per-chunk motion magnitude, capture ratio, direction cosine."""
    pred, gt = npz["pred"], npz["gt"]                     # (Nchunks, H, 48)
    pd = pred[:, :, aa:bb] - pred[:, :1, aa:bb]           # pred displacement from chunk start
    gd = gt[:, :, aa:bb] - gt[:, :1, aa:bb]
    pdn = np.linalg.norm(pd, axis=2)                      # (N,H)
    gdn = np.linalg.norm(gd, axis=2)
    gt_motion = gdn.mean(1)                               # per-chunk motion magnitude
    order = np.argsort(-gt_motion)
    hi = order[: max(1, int(len(order) * hi_frac))]       # high-motion chunks
    # magnitude capture (sum over horizons to be robust)
    ratio = pdn[hi].sum(1) / (gdn[hi].sum(1) + 1e-9)
    # direction cosine on horizons with real motion
    cos = []
    for c in hi:
        m = gdn[c] > 1e-4
        if m.any():
            cs = (pd[c][m] * gd[c][m]).sum(1) / (np.linalg.norm(pd[c][m], axis=1) * gdn[c][m] + 1e-9)
            cos.append(cs.mean())
    # per-dim signed capture: mean over hi chunks of (pred_delta / gt_delta) per dim at final horizon
    pdf = (pred[hi, -1, aa:bb] - pred[hi, 0, aa:bb])
    gdf = (gt[hi, -1, aa:bb] - gt[hi, 0, aa:bb])
    dim_ratio = np.abs(pdf).mean(0) / (np.abs(gdf).mean(0) + 1e-9)
    return dict(n_hi=len(hi), hi_gt_motion=float(gt_motion[hi].mean()),
                ratio_med=float(np.median(ratio)), ratio_mean=float(ratio.mean()),
                dir_cos=float(np.mean(cos)) if cos else float("nan"),
                dim_ratio=dim_ratio)

def main(dirs):
    print(f"{'ckpt':>6} {'segment':22} {'reanch<stay?':>12} {'capture(med)':>12} {'dir_cos':>8} {'n_hi':>5}")
    print("-" * 74)
    for d in dirs:
        step = "".join(ch for ch in os.path.basename(d) if ch.isdigit())
        jp = os.path.join(d, "action_accuracy.json"); npp = os.path.join(d, "all_traces.npz")
        if not os.path.exists(jp):
            print(f"{step:>6} (no result)"); continue
        summ = json.load(open(jp))["summary"]
        npz = np.load(npp) if os.path.exists(npp) else None
        for n, aa, bb in SEGMENTS:
            s = summ[n]
            # high-motion reanchored vs stay from per-sample rows
            rows = [(r[f"{n}_stay_mae"], r.get(f"{n}_reanchored_mae")) for r in json.load(open(jp))["samples"]
                    if r.get(f"{n}_reanchored_mae") is not None]
            rows.sort(key=lambda x: -x[0]); hi = rows[: max(1, len(rows)//3)]
            stay = np.mean([r[0] for r in hi]); rea = np.mean([r[1] for r in hi])
            verdict = f"{'WIN' if rea<stay else 'lose'} {(stay-rea)/stay*100:+.0f}%"
            ms = motion_stats(npz, aa, bb) if npz is not None else {}
            cap = f"{ms.get('ratio_med',float('nan')):.2f}x" if ms else "-"
            dc = f"{ms.get('dir_cos',float('nan')):+.2f}" if ms else "-"
            nh = ms.get('n_hi','-')
            print(f"{step:>6} {n:22} {verdict:>12} {cap:>12} {dc:>8} {str(nh):>5}")
        # per-dim ratios for arms (the manipulation-critical dims)
        if npz is not None:
            for n, aa, bb in SEGMENTS[:2]:
                dr = motion_stats(npz, aa, bb)["dim_ratio"]
                print(f"       {n} per-joint |pred/gt| disp ratio: {np.round(dr,2)}")
        print()

if __name__ == "__main__":
    main(sys.argv[1:])
