#!/usr/bin/env python3
"""Action MAE vs prediction-horizon-step curve, from action_eval.py's all_traces.npz.

action_eval.py dumps all_traces.npz {pred:(N,24,48), gt:(N,24,48), state:(N,48), meta}
where pred/gt are ABSOLUTE joint targets (radians) and state is the current joint state at
the chunk start. This aggregates the per-sample chunks into a mean-|pred-gt| curve as a
function of the horizon step h=1..24, per joint segment (L/R-arm, L/R-hand), overlaying any
number of labelled runs (e.g. trained vs pretraining-only) plus the stay-still floor
(|state-gt| broadcast across the horizon -- the "do nothing" reference action_eval uses).

Usage:
  action_horizon_curve.py --out-png OUT.png --out-json OUT.json \
      trained=output/action_eval_motion2_1500/trained \
      untrained=output/action_eval_motion2_1500/untrained
Each positional arg is label=DIR where DIR contains all_traces.npz.
"""
import argparse, json, os, sys
import numpy as np

SEGMENTS = [
    ("left_arm_joint_pos", 0, 7),
    ("right_arm_joint_pos", 7, 14),
    ("left_hand_joint_pos", 14, 31),
    ("right_hand_joint_pos", 31, 48),
]


def seg_mae_per_step(err_abs):
    """err_abs:(N,H,48) -> {seg: (H,) mean abs error over samples & joints in segment}."""
    out = {}
    for n, a, b in SEGMENTS:
        out[n] = err_abs[:, :, a:b].mean(axis=(0, 2))  # (H,)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=DIR (DIR has all_traces.npz)")
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--stay-from", default=None,
                    help="label whose traces define the stay-still floor (default: first run)")
    a = ap.parse_args()

    runs = []
    for spec in a.runs:
        label, _, d = spec.partition("=")
        npz = os.path.join(d, "all_traces.npz")
        if not os.path.isfile(npz):
            print(f"WARN: no all_traces.npz in {d}; skipping {label}", file=sys.stderr)
            continue
        z = np.load(npz, allow_pickle=True)
        pred, gt = z["pred"].astype(np.float64), z["gt"].astype(np.float64)
        H = gt.shape[1]
        err = np.abs(pred - gt)                       # (N,H,48)
        state = z["state"].astype(np.float64) if "state" in z.files else None
        stay = np.abs(state[:, None, :] - gt) if state is not None else None  # (N,H,48)
        runs.append(dict(label=label, n=int(gt.shape[0]), H=H,
                         policy=seg_mae_per_step(err),
                         stay=seg_mae_per_step(stay) if stay is not None else None))
        print(f"{label}: {gt.shape[0]} chunks, horizon {H}")

    if not runs:
        print("ERROR: no runs with traces", file=sys.stderr); sys.exit(1)

    stay_src = next((r for r in runs if (a.stay_from in (None, r["label"])) and r["stay"]), None)

    # JSON dump (curves are 1-indexed horizon steps for the report)
    out = {"horizon_steps": list(range(1, runs[0]["H"] + 1)), "runs": {}}
    for r in runs:
        out["runs"][r["label"]] = {"n_chunks": r["n"],
                                   "policy_mae": {n: r["policy"][n].tolist() for n, _, _ in SEGMENTS}}
    if stay_src:
        out["stay_still_mae"] = {n: stay_src["stay"][n].tolist() for n, _, _ in SEGMENTS}
    with open(a.out_json, "w") as f:
        json.dump(out, f, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ax, (n, _, _) in zip(axes.ravel(), SEGMENTS):
            steps = np.arange(1, runs[0]["H"] + 1)
            for r in runs:
                ax.plot(steps, r["policy"][n], "-o", ms=3, label=r["label"])
            if stay_src:
                ax.plot(steps, stay_src["stay"][n], "k--", lw=1, label="stay-still floor")
            ax.set_title(n.replace("_joint_pos", "")); ax.grid(True, alpha=.3)
            ax.set_xlabel("horizon step"); ax.set_ylabel("MAE (rad)")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Action MAE vs prediction horizon (lower = better; below stay-still = learned motion)")
        fig.tight_layout(); fig.savefig(a.out_png, dpi=120)
        print(f"wrote {a.out_png}")
    except Exception as e:
        print(f"plot skipped: {e}", file=sys.stderr)
    print(f"wrote {a.out_json}")


if __name__ == "__main__":
    main()
