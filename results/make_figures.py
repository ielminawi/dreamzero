#!/usr/bin/env python
"""Generate results figures from results/data/*.csv. Run: .venv/bin/python results/make_figures.py"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/cluster/scratch/rjiang/dreamzero"
FIG = os.path.join(REPO, "results/figures")
os.makedirs(FIG, exist_ok=True)
df = pd.read_csv(os.path.join(REPO, "results/data/openloop_metrics.csv"))
df = df.drop_duplicates(subset=["run","step","segment"], keep="last")
SEGS = ["left_arm_joint_pos","right_arm_joint_pos","left_hand_joint_pos","right_hand_joint_pos"]
LAB = {s: s.replace("_joint_pos","").replace("_"," ") for s in SEGS}
RUNS = ["varnorm","motion_v1","motion_v2_handspush"]
COL = {"varnorm":"#1f77b4","motion_v1":"#ff7f0e","motion_v2_handspush":"#2ca02c"}

# --- Fig 1: within-chunk motion DIRECTION (dir_cos) vs training step, per limb ---
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
for ax, seg in zip(axes, SEGS):
    for run in RUNS:
        d = df[(df.run==run)&(df.segment==seg)].dropna(subset=["hi_dir_cos"]).sort_values("step")
        if len(d): ax.plot(d.step, d.hi_dir_cos, "o-", color=COL[run], label=run, lw=2, ms=6)
    ax.axhline(0.0, color="gray", ls=":", lw=1)
    ax.set_title(LAB[seg]); ax.set_xlabel("training step"); ax.grid(alpha=0.3)
axes[0].set_ylabel("within-chunk motion direction cosine\n(0 = random, 1 = perfect)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Within-chunk motion is learned ONLY for the right arm; left arm + both hands stay near-random",
             fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig1_dir_cos_trend.png"), dpi=130); plt.close(fig)

# --- Fig 2: per-limb capture ratio + direction at the strongest checkpoint (motion_v1 step 1500) ---
best = df[(df.run=="motion_v1")&(df.step==1500)].set_index("segment")
x = np.arange(len(SEGS)); w=0.38
fig, ax = plt.subplots(figsize=(8.5,4.6))
cap = [best.loc[s,"hi_capture_ratio"] for s in SEGS]
dc  = [best.loc[s,"hi_dir_cos"] for s in SEGS]
b1=ax.bar(x-w/2, cap, w, label="motion-capture ratio  ||pred|| / ||gt||", color="#4c72b0")
b2=ax.bar(x+w/2, dc, w, label="direction cosine", color="#dd8452")
ax.axhline(1.0, color="#4c72b0", ls=":", lw=1); ax.axhline(0.0,color="gray",lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([LAB[s] for s in SEGS]); ax.set_ylabel("value")
ax.set_title("Best model (motion_v1, step 1500): hands under-predict motion (capture~0.1-0.6)\nand move in near-random directions (cos~0.1); right arm is the exception")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{b.get_height():.2f}", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig2_capture_direction_bestckpt.png"), dpi=130); plt.close(fig)

# --- Fig 3: closed-loop task success (object displacement / lift) ---
cl_path = os.path.join(REPO,"results/data/closedloop_task.csv")
if os.path.exists(cl_path):
    cl = pd.read_csv(cl_path)
    # use the most recent sim_run with object logging; average over episodes per object
    run_id = sorted(cl.sim_run.unique())[-1]
    sub = cl[cl.sim_run==run_id].groupby("object").agg(moved_cm=("moved_cm","mean"), max_lift_cm=("max_lift_cm","mean")).reset_index()
    sub["moved_cm"]=sub["moved_cm"].clip(upper=60)  # cap the 6.5m "launched off table" for readability
    fig, ax = plt.subplots(figsize=(8.5,4.4))
    x=np.arange(len(sub)); w=0.38
    ax.bar(x-w/2, sub.moved_cm, w, label="net displacement (cm, capped 60)", color="#c44e52")
    ax.bar(x+w/2, sub.max_lift_cm, w, label="max lift (cm)", color="#55a868")
    ax.set_xticks(x); ax.set_xticklabels(sub.object, rotation=15, fontsize=9); ax.set_ylabel("cm")
    ax.set_title(f"Closed-loop task ({run_id}): objects are SHOVED/knocked but never LIFTED\n(max lift < 3.5cm = no grasp); task not completed")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig3_closedloop_task.png"), dpi=130); plt.close(fig)

print("wrote figures:", sorted(os.listdir(FIG)))
