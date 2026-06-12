"""Render the FrankaOrcaBimanual rig with EVERY joint commanded to exactly 0
(both Franka arms + both Orca hands), from 4 overview angles + the 2 robot
cameras. The all-zero configuration is the kinematic ground truth of the URDF:
comparing it against the official Franka zero pose (and the expected hand
mounting) makes joint-convention / offset errors directly visible, independent
of any recorded data and of the real->sim arm remap.

NOTE: sim panda_joint4 has limits [-3.0718, -0.0698] (a real Panda elbow can
never fully straighten), so a commanded 0 settles at -0.0698 (~4 deg bend).
Every other joint can reach 0 exactly. Achieved values are logged.

Run inside the Isaac container:
  /opt/IsaacLab/isaaclab.sh -p eval_utils/snapshot_zero_pose.py \
      --headless --enable_cameras --device cpu
"""

import argparse

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=str, default="/output/zero_pose")
parser.add_argument("--settle", type=int, default=40,
                    help="settle/camera-warmup steps before capture")
parser.add_argument("--ov-w", type=int, default=1920, help="overview camera width")
parser.add_argument("--ov-h", type=int, default=1440, help="overview camera height")
parser.add_argument("--focus", choices=["none", "left", "right"], default="none",
                    help="aim all overview cameras as CLOSE-UPS on one hand-mount instead of the whole rig")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys
sys.path.insert(0, "/app")
import cv2

import omni.kit.app
_em = omni.kit.app.get_app().get_extension_manager()
for _ext in ["isaacsim.asset.importer.urdf", "omni.importer.urdf", "omni.isaac.urdf"]:
    if _em.set_extension_enabled_immediate(_ext, True):
        break

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

import sim_envs  # noqa: F401  (registers the env)
from sim_envs.franka_orca_bimanual_cfg import (
    FrankaOrcaBimanualEnvCfg,
    LEFT_ARM_ORDER, RIGHT_ARM_ORDER, LEFT_HAND_ORDER, RIGHT_HAND_ORDER,
)
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv

os.makedirs(args.out_dir, exist_ok=True)


def log(*a):
    print(" ".join(str(x) for x in a), flush=True)


def label(img, text, color=(255, 255, 255), scale=0.8):
    out = np.ascontiguousarray(img)
    h = int(34 * scale / 0.8)
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (8, int(h * 0.78)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return out


def save_rgb(path, rgb, text=None, color=(255, 255, 255)):
    img = rgb[..., :3].astype(np.uint8)
    if text is not None:
        img = label(img, text, color)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img


def fit(img, w, h):
    return cv2.resize(img[..., :3].astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA)


# Overview cameras: (name, eye, target, focal_mm). At all-zero the Franka stands
# fully upright (~1.2 m column with the hand on top), so unlike the workspace
# snapshots these aim at z~0.55 and stand farther back to keep the full height in
# frame. The backdrop wall is at x=1.5 -> +X cameras stay at x<1.45.
OVERVIEWS = [
    ("front", (1.40, 0.0, 0.90), (0.0, 0.0, 0.55), 12.0),    # +X looking -X: both arms L|R
    ("top",   (0.05, 0.0, 2.60), (0.10, 0.0, 0.05), 22.0),   # near-straight-down: base yaw / hand yaw
    ("side",  (0.0, -2.00, 0.80), (0.0, 0.0, 0.55), 12.0),   # -Y looking +Y: height profile
    ("persp", (1.35, -1.30, 1.30), (0.0, 0.0, 0.50), 12.0),  # iso
]

# Close-up mode: tight orbit around ONE hand-mount (flange/Gavin-mount/hand-root sit
# near x~0.08, |y|~0.31-0.36, z~0.85-0.92 at the zero pose). 's' flips Y for the side.
if args.focus != "none":
    s = 1.0 if args.focus == "left" else -1.0
    T = (0.08, 0.33 * s, 0.88)  # look-at: the mount/hand assembly
    OVERVIEWS = [
        ("front", (0.85, T[1] + 0.02 * s, 0.93), T, 30.0),          # from +X (workspace side)
        ("top",   (T[0] + 0.01, T[1], 1.75), (T[0], T[1], 0.86), 30.0),
        ("side",  (T[0], T[1] + 0.70 * s, 0.92), T, 30.0),          # from outboard
        ("persp", (0.55, T[1] + 0.45 * s, 1.30), T, 24.0),
    ]


def main():
    log("=" * 72)
    log("ZERO-POSE SNAPSHOT: every joint of both arm+hand articulations -> 0")
    log("=" * 72)

    cfg = FrankaOrcaBimanualEnvCfg()
    cfg.sim.device = args.device
    for name, eye, target, focal in OVERVIEWS:
        setattr(cfg.scene, f"ov_{name}", CameraCfg(
            prim_path=f"/World/OverviewCams/{name}",
            update_period=0.0,
            height=args.ov_h, width=args.ov_w,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal, horizontal_aperture=20.955,
            ),
            offset=CameraCfg.OffsetCfg(pos=eye, rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        ))
    env = FrankaOrcaBimanualEnv(cfg)
    env.reset(); env.reset()

    for name, eye, target, focal in OVERVIEWS:
        cam = env.scene[f"ov_{name}"]
        eyes = torch.tensor([eye], dtype=torch.float32, device=env.device)
        tgts = torch.tensor([target], dtype=torch.float32, device=env.device)
        cam.set_world_poses_from_view(eyes, tgts)
    log("aimed overview cameras via look-at")

    # ---- pin every joint (7 arm + 17 hand per side) at exactly 0, hold target 0 ----
    sides = [
        ("left_arm_hand", LEFT_ARM_ORDER + LEFT_HAND_ORDER),
        ("right_arm_hand", RIGHT_ARM_ORDER + RIGHT_HAND_ORDER),
    ]
    joint_ids = {}
    for skey, names in sides:
        art = env.scene[skey]
        ids, resolved = art.find_joints(list(names), preserve_order=True)
        assert list(resolved) == list(names)
        joint_ids[skey] = ids
        zeros = torch.zeros((1, len(names)), dtype=torch.float32, device=env.device)
        art.write_joint_state_to_sim(zeros, torch.zeros_like(zeros), joint_ids=ids)
        art.set_joint_position_target(zeros, joint_ids=ids)
    env.scene.write_data_to_sim()

    a_zero = torch.zeros((1, 48), dtype=torch.float32, device=env.device)
    for _ in range(max(1, args.settle)):
        env.step(a_zero)

    # ---- report achieved joint values (shows limit clamps, e.g. panda_joint4) ----
    for skey, names in sides:
        art = env.scene[skey]
        q = art.data.joint_pos[0, joint_ids[skey]].cpu().numpy()
        off = [(n, round(float(v), 4)) for n, v in zip(names, q) if abs(v) > 5e-3]
        log(f"{skey}: achieved joints != 0 (commanded 0): {off if off else 'none'}")
        # flange/mount/hand world poses anchor the kinematic chain in world frame
        bn = list(art.body_names)
        for cand in ("panda_link8", "left_gavin_mount", "right_gavin_mount",
                     "left_root", "right_root", "left_palm", "right_palm"):
            if cand in bn:
                p = art.data.body_pos_w[0, bn.index(cand)].cpu().numpy()
                qt = art.data.body_quat_w[0, bn.index(cand)].cpu().numpy()
                log(f"  {skey}: {cand:18s} world pos = {np.round(p, 4)}  quat(wxyz) = {np.round(qt, 4)}")

    # ---- capture ----
    sim_aria = env.scene["aria_rgb_cam"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    sim_oakd = env.scene["oakd_front_view"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    ov_imgs = {}
    for name, *_ in OVERVIEWS:
        ov_imgs[name] = env.scene[f"ov_{name}"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)

    save_rgb(os.path.join(args.out_dir, "zero_aria.png"), sim_aria, "SIM aria  ALL JOINTS = 0", (120, 200, 255))
    save_rgb(os.path.join(args.out_dir, "zero_oakd.png"), sim_oakd, "SIM oak-d  ALL JOINTS = 0", (120, 200, 255))
    for name, *_ in OVERVIEWS:
        save_rgb(os.path.join(args.out_dir, f"zero_{name}.png"), ov_imgs[name],
                 f"ALL JOINTS = 0  {name}  (left arm at +Y = image varies per view)", (255, 230, 120))

    # ---- montage: 4 overviews on top, aria|oakd below ----
    TW = 640
    row1 = np.concatenate([
        label(fit(ov_imgs[n], TW, int(TW * args.ov_h / args.ov_w)), n, (255, 230, 120))
        for n, *_ in OVERVIEWS], axis=1)
    pair = np.concatenate([
        label(fit(sim_aria, TW, 480), "aria (robot cam)", (120, 200, 255)),
        label(fit(sim_oakd, TW, 480), "oakd (robot cam)", (120, 200, 255)),
    ], axis=1)
    pair = cv2.resize(pair, (row1.shape[1], int(pair.shape[0] * row1.shape[1] / pair.shape[1])))
    montage = np.concatenate([row1, pair], axis=0)
    montage = label(montage, "ZERO POSE: all 48 joints commanded to 0 (j4 clamps to -0.0698)", (255, 255, 255), 1.0)
    cv2.imwrite(os.path.join(args.out_dir, "montage_zero_pose.png"),
                cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    log("wrote", os.path.join(args.out_dir, "montage_zero_pose.png"))

    log("=" * 72); log("ZERO-POSE SNAPSHOT COMPLETE"); log("=" * 72)
    env.close(); simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("\n*** EXCEPTION ***"); log(traceback.format_exc()); os._exit(1)
