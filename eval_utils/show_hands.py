"""Render the FrankaOrcaBimanual hands clearly to confirm the fingers/wrists are not
twisted. Uses the sim's VALID home arm pose (so the hands stay in the camera frame) and
commands BOTH hands the SAME finger grasp — a correct (un-twisted) left hand should look
like a mirror image of the right. Renders open vs grasp, full frames + zoomed hand crops.
(Does NOT replay recorded arm actions — those exceed the sim Franka joint4 range and clamp
the arms out of view; that arm joint-convention mismatch is a separate finding.)
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
import cv2

import omni.kit.app
em = omni.kit.app.get_app().get_extension_manager()
for ext in ["isaacsim.asset.importer.urdf", "omni.importer.urdf", "omni.isaac.urdf"]:
    if em.set_extension_enabled_immediate(ext, True):
        break

import sim_envs  # noqa: F401
from sim_envs.franka_orca_bimanual_cfg import (
    FrankaOrcaBimanualEnvCfg, LEFT_HAND_ORDER,
    get_left_arm_joint_pos, get_right_arm_joint_pos,
)
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv

# grasp pose in LEFT_HAND_ORDER: wrist, thumb{mcp,abd,pip,dip}, then index/mid/ring/pinky {abd,mcp,pip}
GRASP = np.array([0.0,  0.5, 0.2, 0.6, 0.6,  0.0, 0.9, 0.8,  0.0, 0.9, 0.8,
                  0.0, 0.9, 0.8,  0.0, 0.9, 0.8], dtype=np.float32)
OPEN = np.zeros(17, dtype=np.float32)


def main():
    cfg = FrankaOrcaBimanualEnvCfg()
    cfg.sim.device = args.device
    env = FrankaOrcaBimanualEnv(cfg)
    env.reset(); env.reset()

    la = get_left_arm_joint_pos(env)[0].cpu().numpy()
    ra = get_right_arm_joint_pos(env)[0].cpu().numpy()

    def render(hand_vec, tag):
        act = torch.tensor(np.concatenate([la, ra, hand_vec, hand_vec]),
                           dtype=torch.float32, device=env.device).unsqueeze(0)
        for _ in range(40):
            obs, _ = env.step(act)
        aria = obs["policy"]["aria_rgb_cam"][0, ..., :3].cpu().numpy().astype(np.uint8)
        oakd = obs["policy"]["oakd_front_view"][0, ..., :3].cpu().numpy().astype(np.uint8)
        cv2.imwrite(f"/output/hands_{tag}_aria.png", cv2.cvtColor(aria, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"/output/hands_{tag}_oakd.png", cv2.cvtColor(oakd, cv2.COLOR_RGB2BGR))
        return aria, oakd

    a_open, o_open = render(OPEN, "open")
    a_grasp, o_grasp = render(GRASP, "grasp")

    # report hand-link world positions so we know where the hands are in the frame
    for key in ("left_arm_hand", "right_arm_hand"):
        art = env.scene[key]
        bn = list(art.body_names)
        # find a fingertip/palm body to report
        for cand in ("left_palm", "right_palm", "left_root", "right_root", "panda_link8"):
            if cand in bn:
                p = art.data.body_pos_w[0, bn.index(cand)].cpu().numpy()
                print(f"{key}: {cand} world pos = {np.round(p,3)}", flush=True)
                break

    # montage: aria(open|grasp) over oakd(open|grasp), labelled
    def lab(im, t):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.putText(im, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return im
    H = 360
    def fit(im):
        s = H / im.shape[0]; return cv2.resize(im, (int(im.shape[1] * s), H))
    row1 = cv2.hconcat([fit(lab(a_open, "aria OPEN")), fit(lab(a_grasp, "aria GRASP"))])
    row2 = cv2.hconcat([fit(lab(o_open, "oakd OPEN")), fit(lab(o_grasp, "oakd GRASP"))])
    w = max(row1.shape[1], row2.shape[1])
    pad = lambda im: cv2.copyMakeBorder(im, 0, 0, 0, w - im.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.imwrite("/output/show_hands.png", cv2.vconcat([pad(row1), pad(row2)]))
    print("wrote /output/show_hands.png + per-config aria/oakd frames", flush=True)
    env.close(); simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback, os
        print("*** EXCEPTION ***\n" + traceback.format_exc(), flush=True); os._exit(1)
