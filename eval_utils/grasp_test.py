"""Scripted grasp test: can the Orca hand actually GRIP and HOLD an object?

Independent of the policy. Places a small object at the right hand's grasp center, closes
the fingers, settles under gravity, then lifts the hand — and checks the object is held
(moves with the hand) vs slips (falls to the table). Validates the friction/grip fix.
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
    FrankaOrcaBimanualEnvCfg, get_right_arm_joint_pos, get_left_arm_joint_pos,
)
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv

_fh = open("/output/grasp_log.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _fh.write(s + "\n"); _fh.flush()

# hand grasp pose (LEFT_HAND_ORDER): wrist, thumb{mcp,abd,pip,dip}, finger{abd,mcp,pip}x4
HALF  = np.array([0.0, 0.4,0.2,0.4,0.4, 0.0,0.5,0.5, 0.0,0.5,0.5, 0.0,0.5,0.5, 0.0,0.5,0.5], dtype=np.float32)
CLOSE = np.array([0.0, 0.7,0.3,1.1,1.1, 0.0,1.3,1.1, 0.0,1.3,1.1, 0.0,1.3,1.1, 0.0,1.3,1.1], dtype=np.float32)
OPEN  = np.zeros(17, dtype=np.float32)


def snap(env, tag):
    oakd = env.scene["oakd_front_view"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    aria = env.scene["aria_rgb_cam"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    return aria, oakd


def main():
    cfg = FrankaOrcaBimanualEnvCfg()
    cfg.sim.device = args.device
    env = FrankaOrcaBimanualEnv(cfg)
    env.reset(); env.reset()

    right = env.scene["right_arm_hand"]
    obj = env.scene["round_item"]            # small sphere (r=0.026), most graspable
    bn = list(right.body_names)
    # distal phalanges of the right hand (where the grip closes)
    tips = [i for i, n in enumerate(bn) if n.startswith("right_") and (n.endswith("_ip") or n.endswith("_dp"))]
    log(f"right-hand distal phalanx bodies used for grasp center: {[bn[i] for i in tips]}")

    la = get_left_arm_joint_pos(env)[0].cpu().numpy()
    ra = get_right_arm_joint_pos(env)[0].cpu().numpy()

    VID = []  # record oak-d frames through the whole sequence -> mp4

    def step_hold(rhand, n, rarm=None):
        arm = ra if rarm is None else rarm
        act = torch.tensor(np.concatenate([la, arm, OPEN, rhand]), dtype=torch.float32,
                           device=env.device).unsqueeze(0)
        for _ in range(n):
            env.step(act)
            VID.append(env.scene["oakd_front_view"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))

    # 1. half-close so fingers form a cage, then read the grasp center
    step_hold(HALF, 40)
    center = right.data.body_pos_w[0, tips].mean(0).cpu().numpy()
    log(f"grasp center (world) = {np.round(center,3)}")

    # 2. teleport the sphere into the grasp center, zero its velocity
    pose = torch.tensor([[*center, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=env.device)
    obj.write_root_pose_to_sim(pose)
    obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    for _ in range(2):
        env.step(torch.tensor(np.concatenate([la, ra, OPEN, HALF]), dtype=torch.float32, device=env.device).unsqueeze(0))
    p_start = obj.data.root_pos_w[0].cpu().numpy()
    a0, o0 = snap(env, "placed")
    log(f"object placed at z={p_start[2]:.3f} (table top=0.05)")

    # 3. close the grip firmly, settle under gravity
    step_hold(CLOSE, 80)
    p_hold = obj.data.root_pos_w[0].cpu().numpy()
    a1, o1 = snap(env, "gripped")
    held_gravity = (p_hold[2] > 0.05 + 0.04) and (np.linalg.norm(p_hold[:2] - center[:2]) < 0.08)
    log(f"after gripping: obj z={p_hold[2]:.3f} dxy_from_center={np.linalg.norm(p_hold[:2]-center[:2]):.3f} "
        f"-> {'HELD (not dropped)' if held_gravity else 'DROPPED/SLIPPED'}")

    # 4. LIFT: raise the hand (shoulder joint up), keep grip closed; object should follow
    ra_lift = ra.copy(); ra_lift[1] -= 0.35  # joint2 (shoulder) — lift the hand
    step_hold(CLOSE, 80, rarm=ra_lift)
    p_lift = obj.data.root_pos_w[0].cpu().numpy()
    hand_now = right.data.body_pos_w[0, tips].mean(0).cpu().numpy()
    a2, o2 = snap(env, "lifted")
    lifted = (p_lift[2] > p_hold[2] + 0.03) and (np.linalg.norm(p_lift - hand_now) < 0.10)
    log(f"after lift: obj z={p_lift[2]:.3f} (was {p_hold[2]:.3f}); dist obj<->hand={np.linalg.norm(p_lift-hand_now):.3f} "
        f"-> {'LIFTED WITH HAND (grasp works!)' if lifted else 'object left behind (grip failed under lift)'}")

    # 5. hold up, then lower back down (still gripped) — makes the video a full pick motion
    step_hold(CLOSE, 30, rarm=ra_lift)
    step_hold(CLOSE, 80)  # lower back to original arm pose

    # ---- montage (placed / gripped / lifted) ----
    def lab(im, t):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR); cv2.putText(im, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2); return im
    H = 360
    fit = lambda im: cv2.resize(im, (int(im.shape[1]*H/im.shape[0]), H))
    row = cv2.hconcat([fit(lab(o0, "placed")), fit(lab(o1, "gripped")), fit(lab(o2, "lifted"))])
    cv2.imwrite("/output/grasp_test.png", row)

    # ---- videos: full oak-d + a zoomed crop on the right hand (top-right region) ----
    if VID:
        h, w = VID[0].shape[:2]
        full = cv2.VideoWriter("/output/grasp_test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        # right hand sits top-right in the top-down oak-d view
        y0, y1, x0, x1 = 0, int(h*0.70), int(w*0.45), w
        zw, zh = (x1-x0)*2, (y1-y0)*2
        zoom = cv2.VideoWriter("/output/grasp_test_zoom.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (zw, zh))
        for fr in VID:
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            full.write(bgr)
            zoom.write(cv2.resize(bgr[y0:y1, x0:x1], (zw, zh)))
        full.release(); zoom.release()
        log(f"wrote /output/grasp_test.mp4 and /output/grasp_test_zoom.mp4 ({len(VID)} frames @30fps)")
    log("wrote /output/grasp_test.png")
    log("RESULT:", "GRASP WORKS" if (held_gravity and lifted) else ("PARTIAL (held but lift failed)" if held_gravity else "GRASP FAILED"))
    env.close(); simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback, os
        log("*** EXCEPTION ***\n" + traceback.format_exc()); _fh.flush(); os._exit(1)
