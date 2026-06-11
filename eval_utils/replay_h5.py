"""Open-loop replay of a recorded bag_groceries episode in the Isaac sim.

Feeds the *recorded* h5 actions (the teleop joint targets) straight into the
FrankaOrcaBimanual env as absolute joint-position targets, in the training action
layout [left_arm(7), right_arm(7), left_hand(17), right_hand(17)]. This isolates the
mechanical chain (action -> joint mapping -> physics, incl. the FINGERS) from the
policy and the visual domain gap: if the sim reproduces the demonstrated motion
(arms reach, fingers curl to grasp), the mechanics are correct and any closed-loop
failure is the policy / domain gap. If the fingers move the wrong way or the arms
don't track, the joint mapping / axis signs / limits are still wrong.

Writes a rollout video + logs commanded-vs-achieved joint tracking (esp. hands).
Run inside the Isaac container (see euler/replay_h5.sbatch).
"""

import argparse
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--h5", type=str,
                    default="/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/20250826_111157.h5")
parser.add_argument("--max-steps", type=int, default=2000)
parser.add_argument("--stride", type=int, default=1, help="subsample recorded frames")
parser.add_argument("--init-to-first", action="store_true", default=True,
                    help="teleport sim joints to recorded action[0] before replay (avoid a startup jump)")
parser.add_argument("--remap-arm", action="store_true", default=True,
                    help="apply configured real->sim arm joint-convention remap")
parser.add_argument("--no-remap-arm", dest="remap_arm", action="store_false")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys
sys.path.insert(0, "/app")
import cv2

import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
for ext_name in ["isaacsim.asset.importer.urdf", "omni.importer.urdf", "omni.isaac.urdf"]:
    if ext_manager.set_extension_enabled_immediate(ext_name, True):
        break

import h5py
import sim_envs  # noqa: F401
from sim_envs.franka_orca_bimanual_cfg import (
    FrankaOrcaBimanualEnvCfg, LEFT_ARM_ORDER, RIGHT_ARM_ORDER, LEFT_HAND_ORDER, RIGHT_HAND_ORDER,
    get_left_hand_joint_pos, get_right_hand_joint_pos, get_left_arm_joint_pos, get_right_arm_joint_pos,
)
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv

LOG = "/output/replay_log.txt"
_fh = open(LOG, "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _fh.write(s + "\n"); _fh.flush()


def main():
    log("=" * 70); log("OPEN-LOOP REPLAY of recorded actions:", args.h5); log("=" * 70)
    with h5py.File(args.h5, "r") as f:
        keys = ["actions_arm_left", "actions_arm_right", "actions_hand_left", "actions_hand_right"]
        if not all(k in f for k in keys):
            log("ERROR: h5 missing bimanual action keys; has:", list(f.keys())); return
        al = f["actions_arm_left"][:]; ar = f["actions_arm_right"][:]
        hl = f["actions_hand_left"][:]; hr = f["actions_hand_right"][:]
        # also grab recorded qpos to compare ranges
        qpos_hl = f["observations/qpos_hand_left"][:] if "observations/qpos_hand_left" in f else None
    T = min(len(al), len(ar), len(hl), len(hr))
    log(f"episode length T={T}; arm{al.shape[1]} hand{hl.shape[1]} per side")
    al = al[:T]; ar = ar[:T]; hl = hl[:T]; hr = hr[:T]
    # Apply the configured arm joint-convention remap (real -> sim).
    # Without this the recorded arm joint4 (~+0.9) is outside the sim limit [-3.07,-0.07]
    # and clamps (arms go out of view). Disable with --no-remap-arm to see the broken case.
    from sim_envs.franka_orca_bimanual_cfg import real_arm_to_sim, ARM_SIM_FROM_REAL
    if args.remap_arm:
        log(f"APPLYING arm remap real->sim: ARM_SIM_FROM_REAL={ARM_SIM_FROM_REAL}")
        al = np.stack([real_arm_to_sim(a, "left") for a in al]).astype(np.float32)
        ar = np.stack([real_arm_to_sim(a, "right") for a in ar]).astype(np.float32)
    else:
        log("NO arm remap (raw recorded actions)")
    # full 48-dim action in TRAINING layout [L_arm, R_arm, L_hand, R_hand]
    actions = np.concatenate([al, ar, hl, hr], axis=-1).astype(np.float32)
    log("recorded action ranges (rad):")
    log(f"  L_arm  min={al.min(0).round(2)} max={al.max(0).round(2)}")
    log(f"  L_hand min={hl.min(0).round(2)}")
    log(f"  L_hand max={hl.max(0).round(2)}")

    cfg = FrankaOrcaBimanualEnvCfg()
    cfg.sim.device = args.device
    env = FrankaOrcaBimanualEnv(cfg)
    env.reset(); env.reset()

    # report sim joint limits for the hand (to see if recorded actions exceed them)
    left = env.scene["left_arm_hand"]
    lims = None
    for attr in ("joint_pos_limits", "joint_limits"):
        if hasattr(left.data, attr) and getattr(left.data, attr) is not None:
            lims = getattr(left.data, attr)[0].cpu().numpy(); break
    names = list(left.joint_names)
    if lims is not None:
        log("\nsim left hand joint limits (training order):")
        cols = {n: i for i, n in enumerate(names)}
        for j, nm in enumerate(LEFT_HAND_ORDER):
            i = cols[nm]; log(f"  {nm:18s} [{lims[i,0]:+.2f}, {lims[i,1]:+.2f}]  recorded[min,max]=[{hl[:,j].min():+.2f},{hl[:,j].max():+.2f}]")

    idxs = list(range(0, T, args.stride))[: args.max_steps]
    video = []
    track_log_at = set(int(len(idxs) * p) for p in (0.25, 0.5, 0.75, 0.99))
    for n, t in enumerate(idxs):
        act = torch.tensor(actions[t], dtype=torch.float32, device=env.device).unsqueeze(0)
        obs, _ = env.step(act)
        oakd = obs["policy"]["oakd_front_view"][0, ..., :3].cpu().numpy().astype(np.uint8)
        aria = obs["policy"]["aria_rgb_cam"][0, ..., :3].cpu().numpy().astype(np.uint8)
        aria_v = cv2.resize(aria, (320, 240)); oakd_v = cv2.resize(oakd, (320, 240))
        video.append(np.concatenate([aria_v, oakd_v], axis=1))
        if n in track_log_at:
            # ARM: commanded (sim convention, already remapped in `actions`) vs achieved.
            # If the remap is right, achieved tracks cmd and is NOT pinned at a joint limit.
            arm_cmd = actions[t, 0:7]
            arm_ach = get_left_arm_joint_pos(env)[0].cpu().numpy()
            aerr = np.abs(arm_ach - arm_cmd)
            log(f"\n[step {t}] L_arm cmd vs achieved (max|err|={aerr.max():.3f})  [j4,j6 are the remapped ones]")
            log(f"  arm_cmd(sim)={np.round(arm_cmd,2)}")
            log(f"  arm_ach     ={np.round(arm_ach,2)}")
            # HAND: commanded vs achieved (training order)
            ach = get_left_hand_joint_pos(env)[0].cpu().numpy()
            cmd = hl[t]
            log(f"  L_hand max|err|={np.abs(ach-cmd).max():.3f}")
    # write video (BGR for cv2)
    out = "/output/replay.mp4"
    h, w = video[0].shape[:2]
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 50, (w, h))
    for fr in video:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    log(f"\nwrote {out} ({len(video)} frames). aria|oakd side-by-side.")
    log("=" * 70); log("REPLAY COMPLETE"); log("=" * 70)
    env.close(); simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback, os
        log("\n*** EXCEPTION ***"); log(traceback.format_exc()); _fh.flush(); os._exit(1)
