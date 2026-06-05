#!/usr/bin/env python3
"""Forward action normalization for action-conditioned WM inference.

Maps raw absolute parquet actions -> the model's NORMALIZED relative action space:
  rel  = action_abs - ref_state          (per segment; same ref for all horizon steps)
  norm = clip( 2*(rel - q01)/(q99 - q01) - 1 , -1, 1 )
using the relative-action stats (meta/relative_stats_dreamzero.json), which is exactly
what the model trained on (relative_action=true, q99). Inverse == sim_policy.unapply.
"""
import json
import numpy as np

# 48-dim action/state layout for franka_orca_bimanual
SEGMENTS = [
    ("left_arm_joint_pos", 0, 7),
    ("right_arm_joint_pos", 7, 14),
    ("left_hand_joint_pos", 14, 31),
    ("right_hand_joint_pos", 31, 48),
]


def load_stats(path):
    with open(path) as f:
        s = json.load(f)
    out = {}
    for name, a, b in SEGMENTS:
        out[name] = (np.asarray(s[name]["q01"], np.float64),
                     np.asarray(s[name]["q99"], np.float64))
    return out


def normalize_action_chunk(action_abs, ref_state, stats):
    """action_abs (T,48) absolute, ref_state (48,) -> normalized (T,48) in [-1,1]."""
    action_abs = np.asarray(action_abs, np.float64)
    ref_state = np.asarray(ref_state, np.float64)
    out = np.empty_like(action_abs)
    for name, a, b in SEGMENTS:
        q01, q99 = stats[name]
        rel = action_abs[:, a:b] - ref_state[a:b]          # broadcast ref over T
        denom = (q99 - q01)
        mask = denom != 0
        norm = rel.copy()
        norm[:, mask] = 2 * (rel[:, mask] - q01[mask]) / denom[mask] - 1
        out[:, a:b] = np.clip(norm, -1.0, 1.0)
    return out


def denormalize_action_chunk(norm, ref_state, stats):
    """Inverse of normalize_action_chunk (== sim_policy.unapply): norm -> absolute."""
    norm = np.asarray(norm, np.float64)
    ref_state = np.asarray(ref_state, np.float64)
    out = np.empty_like(norm)
    for name, a, b in SEGMENTS:
        q01, q99 = stats[name]
        rel = (norm[:, a:b] + 1) / 2 * (q99 - q01) + q01
        out[:, a:b] = rel + ref_state[a:b]
    return out
