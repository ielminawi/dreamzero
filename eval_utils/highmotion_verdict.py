#!/usr/bin/env python
"""Offset-corrected high-motion verdict from an action_accuracy.json.
The summary's highmotion_policy_mae is RAW (includes the per-chunk anchor offset that is
removed at deploy by re-anchoring). The honest deploy metric is the RE-ANCHORED MAE on the
high-motion subset. This recomputes that from the per-sample rows.

Usage: python highmotion_verdict.py <action_accuracy.json> [top_fraction=0.33]
"""
import json, sys, statistics as st

path = sys.argv[1]
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.33
d = json.load(open(path))
S = d["samples"]
groups = ["left_arm_joint_pos", "right_arm_joint_pos", "left_hand_joint_pos", "right_hand_joint_pos"]
print(f"# {path}  (n_samples={len(S)}, top {frac:.0%} by stay-motion = high-motion)")
wins = 0
for g in groups:
    rows = [(s[f"{g}_stay_mae"], s[f"{g}_mae"], s.get(f"{g}_reanchored_mae"))
            for s in S if f"{g}_stay_mae" in s and s.get(f"{g}_reanchored_mae") is not None]
    if not rows:
        print(f"{g:22s} (no reanchored data)"); continue
    rows.sort(key=lambda x: -x[0])
    hm = rows[: max(1, int(len(rows) * frac))]
    stay = st.mean(r[0] for r in hm); raw = st.mean(r[1] for r in hm); rea = st.mean(r[2] for r in hm)
    win = rea < stay
    wins += win
    print(f"{g:22s} HM(n={len(hm):3d}): stay={stay:.4f} raw={raw:.4f} REANCH={rea:.4f} "
          f"-> {'WIN ' if win else 'lose'} ({(stay-rea)/stay*100:+.0f}%)")
print(f"# VERDICT: reanchored beats stay on high-motion in {wins}/4 groups")
