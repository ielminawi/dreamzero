"""Validate the FrankaOrcaBimanual sim joint setup against the training contract.

Answers three questions that decide whether the sim matches training:
  1) Which action path is LIVE — the manager (cfg ActionsCfg/JointPositionActionCfg)
     or the DirectRLEnv-style _apply_action override in the env?
  2) For the live path, how is the 48-dim action applied? (scale/offset/use_default_offset
     for the manager terms.) The DreamZero server returns ABSOLUTE joint targets, so the
     correct application must be target = action (no current- or default-pose offset added).
  3) Does the IsaacLab articulation joint ORDER match the URDF/training order
     [panda_joint1..7, {side}_wrist, thumb_mcp, thumb_abd, thumb_pip, thumb_dip,
      index_abd, index_mcp, index_pip, middle_abd, middle_mcp, middle_pip,
      ring_abd, ring_mcp, ring_pip, pinky_abd, pinky_mcp, pinky_pip]?

Run inside the Isaac container (see euler/inspect_joints.sbatch).
"""

import argparse
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys
sys.path.insert(0, "/app")

import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
for ext_name in ["isaacsim.asset.importer.urdf", "omni.importer.urdf", "omni.isaac.urdf"]:
    if ext_manager.set_extension_enabled_immediate(ext_name, True):
        print(f"[INFO] Enabled extension: {ext_name}")
        break

import sim_envs  # noqa: F401
from sim_envs.franka_orca_bimanual_cfg import FrankaOrcaBimanualEnvCfg
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv

# Isaac captures/buffers Python stdout into its in-container kit log, so the SLURM
# log loses our prints. Mirror everything to a bound output file (/output -> repo/output/sim).
_REPORT_PATH = "/output/joint_inspection.txt"
_report_fh = open(_REPORT_PATH, "w")


def report(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    _report_fh.write(line + "\n")
    _report_fh.flush()


def side_expected(side):
    arm = [f"panda_joint{i}" for i in range(1, 8)]
    hand = [f"{side}_wrist",
            f"{side}_thumb_mcp", f"{side}_thumb_abd", f"{side}_thumb_pip", f"{side}_thumb_dip",
            f"{side}_index_abd", f"{side}_index_mcp", f"{side}_index_pip",
            f"{side}_middle_abd", f"{side}_middle_mcp", f"{side}_middle_pip",
            f"{side}_ring_abd", f"{side}_ring_mcp", f"{side}_ring_pip",
            f"{side}_pinky_abd", f"{side}_pinky_mcp", f"{side}_pinky_pip"]
    return arm + hand


def _summ(x):
    if x is None:
        return "None"
    if torch.is_tensor(x):
        x = x.float()
        return (f"tensor mean={x.mean().item():.4f} min={x.min().item():.4f} "
                f"max={x.max().item():.4f} shape={tuple(x.shape)} vals={x.flatten()[:8].tolist()}")
    return repr(x)


def main():
    report("\n" + "=" * 70)
    report("FrankaOrcaBimanual JOINT SETUP VALIDATION")
    report("=" * 70)

    cfg = FrankaOrcaBimanualEnvCfg()
    cfg.sim.device = args.device
    env = FrankaOrcaBimanualEnv(cfg)
    env.reset()
    env.reset()

    # ---- (1) which action path is live? ----
    report("\n[1] LIVE ACTION PATH")
    am = getattr(env, "action_manager", None)
    if am is not None and len(getattr(am, "active_terms", [])) > 0:
        report("  -> ManagerBasedEnv ActionManager is LIVE (env._apply_action is dead code).")
        report("     active action terms (THIS is the action concat order):", am.active_terms)
        report("     total action dim:", am.total_action_dim)
        report("     action_term_dim:", getattr(am, "action_term_dim", None))
    else:
        report("  -> No active ActionManager; the env's _apply_action override is the live path.")

    # ---- (2) per-term application semantics (manager path) ----
    report("\n[2] ACTION TERM SEMANTICS (server returns ABSOLUTE targets)")
    if am is not None:
        for name in am.active_terms:
            term = am.get_term(name)
            report(f"  term '{name}': action_dim={term.action_dim}")
            report(f"     joint_names (term apply order) = {getattr(term, '_joint_names', None)}")
            report(f"     joint_ids = {getattr(term, '_joint_ids', None)}")
            report(f"     scale  = {_summ(getattr(term, '_scale', None))}")
            report(f"     offset = {_summ(getattr(term, '_offset', None))}")
            report("       (NONZERO offset => default-pose added => WRONG for absolute targets)")

    # ---- (3) articulation joint ORDER vs training/URDF ----
    report("\n[3] ARTICULATION JOINT ORDER vs TRAINING/URDF ORDER")
    for scene_key, side in [("left_arm_hand", "left"), ("right_arm_hand", "right")]:
        art = env.scene[scene_key]
        actual = list(art.joint_names)
        expected = side_expected(side)
        report(f"\n  --- {scene_key} ({len(actual)} joints) ---")
        report(f"  actual  : {actual}")
        report(f"  expected: {expected}")
        if actual == expected:
            report("  RESULT: MATCH (raw slicing [:, :7] / [:, 7:24] is CORRECT)")
        else:
            report("  RESULT: ***MISMATCH*** -- Isaac reordered joints. Raw slicing is WRONG.")
            remap = [actual.index(j) if j in actual else -1 for j in expected]
            report(f"  remap (expected_idx -> articulation_idx): {remap}")
            missing = [j for j in expected if j not in actual]
            extra = [j for j in actual if j not in expected]
            if missing:
                report(f"  MISSING from articulation: {missing}")
            if extra:
                report(f"  EXTRA in articulation: {extra}")

    # ---- (4) round-trip test: feed the observed state back as an ABSOLUTE action ----
    # Build the 48-dim action exactly like the real eval client: concatenate the four
    # observation getters in the SERVER/training layout [left_arm, right_arm, left_hand,
    # right_hand]. The getters return joints in canonical TRAINING order; the action terms
    # consume them in the same order (preserve_order). If actions are applied as pure
    # absolute targets, commanding target == current_state should keep the robot still.
    from sim_envs.franka_orca_bimanual_cfg import (
        get_left_arm_joint_pos, get_right_arm_joint_pos,
        get_left_hand_joint_pos, get_right_hand_joint_pos,
        LEFT_HAND_ORDER,
    )
    report("\n[4] ROUND-TRIP TEST (action = observed state, in training layout)")

    def obs_state():
        # exactly what the eval client sends, in training order
        return (get_left_arm_joint_pos(env)[0].clone(),
                get_right_arm_joint_pos(env)[0].clone(),
                get_left_hand_joint_pos(env)[0].clone(),
                get_right_hand_joint_pos(env)[0].clone())

    la0, ra0, lh0, rh0 = obs_state()
    action = torch.cat([la0, ra0, lh0, rh0]).unsqueeze(0)
    report(f"  action shape = {tuple(action.shape)} (expect (1,48))")

    def drifts():
        la, ra, lh, rh = obs_state()
        return (
            (la - la0).abs().max().item(), (ra - ra0).abs().max().item(),
            (lh - lh0).abs().max().item(), (rh - rh0).abs().max().item(),
        )

    for n in (10, 30):
        for _ in range(n - (10 if n == 30 else 0)):
            env.step(action)
        dla, dra, dlh, drh = drifts()
        report(f"  after {n:>2} steps  max|Δ|:  L_arm={dla:.4f}  R_arm={dra:.4f}  "
               f"L_hand={dlh:.4f}  R_hand={drh:.4f}")
    # per-joint hand drift to localize the droop
    lh = get_left_hand_joint_pos(env)[0]
    perj = [round((lh[i] - lh0[i]).abs().item(), 3) for i in range(lh.shape[0])]
    report(f"  L_hand per-joint |Δ| (training order {LEFT_HAND_ORDER[0]}..): {perj}")
    report("  EXPECT arm drift ~0 (proves absolute targets + correct term order + arm mapping).")
    report("  Hand drift should also be small now that hand gains were raised (stiffness 20,")
    report("  damping 1.0) and the fingertip links have valid inertia.")

    # ---- (5) scene objects: rest ON the table (not sunk through it) + camera frame ----
    report("\n[5] SCENE OBJECTS (bag_groceries stand-ins)")
    TABLE_TOP = 0.05
    # expected resting z = table_top + half-extent (object sitting ON the table).
    # If z ~= half-extent instead, the object sank THROUGH the table to the ground (z=0).
    expect_top = {"grocery_bag": 0.05 + 0.095, "white_container": 0.05 + 0.0225,
                  "blue_tube": 0.05 + 0.018, "round_item": 0.05 + 0.026}
    obj_keys = list(expect_top)
    # settle for ~1s holding the arms at the current pose
    for _ in range(50):
        env.step(action)
    for k in obj_keys:
        try:
            z = env.scene[k].data.root_pos_w[0, 2].item()
        except KeyError:
            report(f"  {k}: NOT in scene"); continue
        exp = expect_top[k]
        on_table = abs(z - exp) < 0.02
        sunk = abs(z - (exp - TABLE_TOP)) < 0.02
        verdict = "ON TABLE" if on_table else ("SUNK THROUGH TABLE" if sunk else "OTHER")
        report(f"  {k:16s} z={z:.3f}  (expect on-table {exp:.3f}, ground {exp-TABLE_TOP:.3f})  {verdict}")
    # save an oak-d camera frame to compare against the training video
    try:
        import cv2
        rgb = env.scene["oakd_front_view"].data.output["rgb"][0, ..., :3].cpu().numpy()
        cv2.imwrite("/output/sim_oakd_frame.png", cv2.cvtColor(rgb.astype("uint8"), cv2.COLOR_RGB2BGR))
        report("  saved sim oak-d frame -> /output/sim_oakd_frame.png")
    except Exception as e:
        report(f"  camera frame save failed: {e}")

    report("\n" + "=" * 70)
    report("VALIDATION COMPLETE")
    report("=" * 70)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback, os
        report("\n*** EXCEPTION during validation ***")
        report(traceback.format_exc())
        # Isaac's simulation_app.close() can hang for many minutes after an error;
        # results are already flushed to the report file, so exit hard.
        _report_fh.flush()
        os._exit(1)
