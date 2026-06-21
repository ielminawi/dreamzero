#!/usr/bin/env python
"""Consolidate all open-loop + closed-loop eval artifacts into tidy CSVs for the results section.
Outputs:
  results/data/openloop_metrics.csv   - per (run, step, segment): summary metrics + high-motion motion-capture/direction
  results/data/closedloop_task.csv    - per (episode, object): displacement + max-lift (task success proxy)
  results/data/openloop_metrics.json  - same open-loop data, nested, for convenience
Run: .venv/bin/python results/consolidate_metrics.py
"""
import json, os, glob, csv
import numpy as np

REPO = "/cluster/scratch/rjiang/dreamzero"
SEGMENTS = [("left_arm_joint_pos", 0, 7), ("right_arm_joint_pos", 7, 14),
            ("left_hand_joint_pos", 14, 31), ("right_hand_joint_pos", 31, 48)]

# (run_label, output_dir_glob, step_from_name_fn). Steps parsed from dir name.
RUNS = [
    ("clean(plateaued baseline)", "output/action_eval_clean_*"),
    ("varnorm", "output/action_eval_varnorm_*_full"),
    ("motion_v1", "output/action_eval_motion_*_full"),
    ("motion_v2_handspush", "output/action_eval_motion2_*_full"),
]

def step_of(dirname, run):
    import re
    base = os.path.basename(dirname)
    # strip the run token so motion2_500 -> 500 (not 2500)
    base = base.replace("motion2", "").replace("reanchor", "")
    m = re.findall(r"(\d+)", base)
    return int(m[-1]) if m else -1

def hi_motion_stats(npz_path, aa, bb, frac=0.33):
    """High-motion subset: capture ratio (||pred_disp||/||gt_disp||) + direction cosine + per-dim ratio."""
    if not os.path.exists(npz_path):
        return {}
    z = np.load(npz_path); pred, gt = z["pred"], z["gt"]
    pd = pred[:, :, aa:bb] - pred[:, :1, aa:bb]
    gd = gt[:, :, aa:bb] - gt[:, :1, aa:bb]
    pdn = np.linalg.norm(pd, axis=2); gdn = np.linalg.norm(gd, axis=2)
    mot = gdn.mean(1); hi = np.argsort(-mot)[: max(1, int(len(mot) * frac))]
    cap = float(pdn[hi].sum() / (gdn[hi].sum() + 1e-9))
    cos = []
    for c in hi:
        m = gdn[c] > 1e-4
        if m.any():
            cos.append(float(((pd[c][m]*gd[c][m]).sum(1)/(np.linalg.norm(pd[c][m],axis=1)*gdn[c][m]+1e-9)).mean()))
    return dict(hi_motion_rad=float(mot[hi].mean()), capture_ratio=cap,
                dir_cos=float(np.mean(cos)) if cos else float("nan"), n_hi=int(len(hi)))

rows = []
nested = {}
for run, pat in RUNS:
    for d in sorted(glob.glob(os.path.join(REPO, pat))):
        jp = os.path.join(d, "action_accuracy.json")
        if not os.path.exists(jp):
            continue
        try:
            summ = json.load(open(jp)).get("summary", {})
        except Exception:
            continue
        if not summ:
            continue
        step = step_of(d, run)
        npz = os.path.join(d, "all_traces.npz")
        for seg, aa, bb in SEGMENTS:
            s = summ.get(seg, {})
            ms = hi_motion_stats(npz, aa, bb)
            row = dict(run=run, step=step, segment=seg,
                       policy_mae=s.get("policy_mae"), stay_mae=s.get("stay_mae"),
                       reanchored_mae=s.get("policy_reanchored_mae"), disp_mae=s.get("policy_disp_mae"),
                       corr=s.get("corr"),
                       hi_capture_ratio=ms.get("capture_ratio"), hi_dir_cos=ms.get("dir_cos"),
                       hi_motion_rad=ms.get("hi_motion_rad"), n_hi=ms.get("n_hi"))
            rows.append(row)
            nested.setdefault(run, {}).setdefault(step, {})[seg] = row

cols = ["run","step","segment","policy_mae","stay_mae","reanchored_mae","disp_mae","corr",
        "hi_capture_ratio","hi_dir_cos","hi_motion_rad","n_hi"]
os.makedirs(os.path.join(REPO,"results/data"), exist_ok=True)
with open(os.path.join(REPO,"results/data/openloop_metrics.csv"),"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in sorted(rows, key=lambda x:(x["run"],x["step"],x["segment"])):
        w.writerow({k: (round(v,4) if isinstance(v,float) else v) for k,v in r.items()})
json.dump(nested, open(os.path.join(REPO,"results/data/openloop_metrics.json"),"w"), indent=2, default=str)
print(f"wrote openloop_metrics.csv: {len(rows)} rows from {len(set((r['run'],r['step']) for r in rows))} checkpoints")

# ---- closed-loop task success (object poses) ----
crows = []
for objnpz in sorted(glob.glob(os.path.join(REPO,"output/sim/2026-06-18/*/episode_*_objects.npz"))):
    run_dir = os.path.basename(os.path.dirname(objnpz))
    ep = os.path.basename(objnpz).split("_")[1]
    z = np.load(objnpz)
    for k in z.files:
        a = z[k]  # (T,7) xyz+quat
        disp_cm = float(np.linalg.norm(a[-1,:3]-a[0,:3])*100)
        lift_cm = float((a[:,2]-a[0,2]).max()*100)
        crows.append(dict(sim_run=run_dir, episode=ep, object=k,
                          moved_cm=round(disp_cm,1), max_lift_cm=round(lift_cm,1)))
with open(os.path.join(REPO,"results/data/closedloop_task.csv"),"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["sim_run","episode","object","moved_cm","max_lift_cm"]); w.writeheader()
    for r in crows: w.writerow(r)
print(f"wrote closedloop_task.csv: {len(crows)} object-episode rows")
