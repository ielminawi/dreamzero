"""Render the sim aria + oak-d cameras at a real demo pose, to tune them to match the
training videos. Applies a recorded h5 action (arms remapped real->sim so they reach the
demo pose) at a chosen step, settles, and saves full-res aria + oak-d frames. Compare
against the training frames at the same step.
"""
import argparse
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--h5", type=str,
                    default="/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/20250826_111157.h5")
parser.add_argument("--steps", type=int, nargs="+", default=[1000, 1600])
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
from sim_envs.franka_orca_bimanual_cfg import FrankaOrcaBimanualEnvCfg, real_arm_to_sim
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv


def main():
    with h5py.File(args.h5, "r") as f:
        al = f["actions_arm_left"][:]; ar = f["actions_arm_right"][:]
        hl = f["actions_hand_left"][:]; hr = f["actions_hand_right"][:]
    T = min(len(al), len(ar), len(hl), len(hr))
    al = np.stack([real_arm_to_sim(a) for a in al[:T]]).astype(np.float32)
    ar = np.stack([real_arm_to_sim(a) for a in ar[:T]]).astype(np.float32)
    actions = np.concatenate([al, ar, hl[:T], hr[:T]], axis=-1).astype(np.float32)

    cfg = FrankaOrcaBimanualEnvCfg()
    cfg.sim.device = args.device
    env = FrankaOrcaBimanualEnv(cfg)
    env.reset(); env.reset()

    for st in args.steps:
        st = min(st, T - 1)
        act = torch.tensor(actions[st], dtype=torch.float32, device=env.device).unsqueeze(0)
        for _ in range(60):
            obs, _ = env.step(act)
        aria = obs["policy"]["aria_rgb_cam"][0, ..., :3].cpu().numpy().astype(np.uint8)
        oakd = obs["policy"]["oakd_front_view"][0, ..., :3].cpu().numpy().astype(np.uint8)
        cv2.imwrite(f"/output/cam_aria_step{st}.png", cv2.cvtColor(aria, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"/output/cam_oakd_step{st}.png", cv2.cvtColor(oakd, cv2.COLOR_RGB2BGR))
        print(f"step {st}: saved cam_aria_step{st}.png ({aria.shape}) + cam_oakd_step{st}.png ({oakd.shape})", flush=True)
    print("DONE", flush=True)
    env.close(); simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback, os
        print("*** EXCEPTION ***\n" + traceback.format_exc(), flush=True); os._exit(1)
