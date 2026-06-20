"""Render ONE aria+oakd frame using the CURRENT cfg (post camera-match edits) at a
recorded pose, so the hands are in view. Used to validate the SIM-vs-train OOD fixes.

Drives both arms to a recorded EE pose (IK'd like match_oakd_camera.py) and writes the
cfg-rendered aria/oakd frames as PNG. No runtime camera override — this tests the cfg.
"""
import argparse
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--h5", type=str,
                    default="/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/20250826_111157.h5")
parser.add_argument("--step", type=int, default=1000)
parser.add_argument("--out", type=str, default="/output")
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
em = omni.kit.app.get_app().get_extension_manager()
for ext in ["isaacsim.asset.importer.urdf", "omni.importer.urdf", "omni.isaac.urdf"]:
    if em.set_extension_enabled_immediate(ext, True):
        break

import h5py
import sim_envs  # noqa: F401
from sim_envs.franka_orca_bimanual_cfg import FrankaOrcaBimanualEnvCfg
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv
from eval_utils import ee_ik


def main():
    with h5py.File(args.h5, "r") as f:
        pl = f["observations/qpos_arm_left"][:]; pr = f["observations/qpos_arm_right"][:]
        hl = f["observations/qpos_hand_left"][:]; hr = f["observations/qpos_hand_right"][:]
    st = min(args.step, min(len(pl), len(pr)) - 1)
    qL, eL, _ = ee_ik.solve_ik(pl[st], ee_ik.Q_HOME)
    qR, eR, _ = ee_ik.solve_ik(pr[st], ee_ik.Q_HOME)
    print(f"step {st}: IK residual L={eL*1000:.1f}mm R={eR*1000:.1f}mm", flush=True)
    act_np = np.concatenate([qL, qR, hl[st], hr[st]]).astype(np.float32)

    cfg = FrankaOrcaBimanualEnvCfg()
    cfg.sim.device = args.device
    env = FrankaOrcaBimanualEnv(cfg)
    env.reset(); env.reset()
    act = torch.tensor(act_np, dtype=torch.float32, device=env.device).unsqueeze(0)
    for _ in range(90):
        obs, _ = env.step(act)

    aria = obs["policy"]["aria_rgb_cam"][0, ..., :3].cpu().numpy().astype(np.uint8)
    oakd = obs["policy"]["oakd_front_view"][0, ..., :3].cpu().numpy().astype(np.uint8)
    cv2.imwrite(f"{args.out}/match_aria_step{st}.png", cv2.cvtColor(aria, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{args.out}/match_oakd_step{st}.png", cv2.cvtColor(oakd, cv2.COLOR_RGB2BGR))
    print(f"aria shape {aria.shape}  oakd shape {oakd.shape}", flush=True)
    print(f"saved {args.out}/match_aria_step{st}.png  {args.out}/match_oakd_step{st}.png", flush=True)
    print("DONE", flush=True)
    env.close(); simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback, os
        print("*** EXCEPTION ***\n" + traceback.format_exc(), flush=True); os._exit(1)
