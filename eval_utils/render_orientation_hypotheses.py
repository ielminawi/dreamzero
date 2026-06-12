"""Render one frame under many orientation-convention hypotheses for eyeballing.

Loads a hypotheses npz from export_orientation_hypotheses.py (ik_left/ik_right per
candidate + labels + residuals), and for EACH candidate teleports the arms to that
exact IK solution (hands held at the recorded qpos_hand of the frame), then captures
overview + robot cameras. Builds ONE contact sheet: a row per candidate
  [ iso | top | SIM aria | REAL aria | SIM oakd | REAL oakd ]
with the candidate label + per-arm IK residual burned in. The REAL columns are the
same recorded frame (the fixed reference). You scan rows for the hand orientation
that matches the real hand.

Run inside the Isaac container (see docker/scripts/run_orientation_hypotheses.sh):
  /opt/IsaacLab/isaaclab.sh -p eval_utils/render_orientation_hypotheses.py \
      --headless --enable_cameras --device cpu --h5 /app/20250826_111157.h5 \
      --npz /app/ori_hyp_flange_t1400.npz --frame 1400 --out-dir /output
"""

import argparse

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--h5", type=str, default="/app/20250826_111157.h5")
parser.add_argument("--npz", type=str, required=True)
parser.add_argument("--frame", type=int, default=1400)
parser.add_argument("--out-dir", type=str, default="/output")
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--settle", type=int, default=12)
parser.add_argument("--ov-w", type=int, default=1600)
parser.add_argument("--ov-h", type=int, default=1200)
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

import h5py
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

import sim_envs  # noqa: F401
from sim_envs.franka_orca_bimanual_cfg import (
    FrankaOrcaBimanualEnvCfg,
    LEFT_ARM_ORDER, RIGHT_ARM_ORDER, LEFT_HAND_ORDER, RIGHT_HAND_ORDER,
)
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv

ARIA_DS = "observations/images/aria_rgb_cam/color"
OAKD_DS = "observations/images/oakd_front_view/color"
os.makedirs(args.out_dir, exist_ok=True)


def log(*a):
    print(" ".join(str(x) for x in a), flush=True)


def label(img, text, color=(255, 255, 255), scale=0.7):
    out = np.ascontiguousarray(img)
    h = max(24, int(30 * scale / 0.7))
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (8, int(h * 0.74)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return out


def fit(img, w, h):
    return cv2.resize(img[..., :3].astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA)


OVERVIEWS = [
    ("iso",  (1.15, -1.15, 0.95), (0.28, 0.0, 0.12), 14.0),
    ("top",  (0.30, 0.0, 2.20),   (0.38, 0.0, 0.05), 26.0),
]


def main():
    npz = np.load(args.npz, allow_pickle=True)
    ikL = npz["ik_left"].astype(np.float32); ikR = npz["ik_right"].astype(np.float32)
    labels = [str(x) for x in npz["labels"]]
    resid = npz["residual"]
    model = str(npz["model"])
    H = len(labels)
    log(f"rendering {H} orientation hypotheses, model={model}, frame={args.frame}")

    cfg = FrankaOrcaBimanualEnvCfg()
    cfg.sim.device = args.device
    for name, eye, target, focal in OVERVIEWS:
        setattr(cfg.scene, f"ov_{name}", CameraCfg(
            prim_path=f"/World/OverviewCams/{name}",
            update_period=0.0, height=args.ov_h, width=args.ov_w,
            spawn=sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=20.955),
            offset=CameraCfg.OffsetCfg(pos=eye, rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        ))
    env = FrankaOrcaBimanualEnv(cfg)
    env.reset(); env.reset()
    for name, eye, target, focal in OVERVIEWS:
        env.scene[f"ov_{name}"].set_world_poses_from_view(
            torch.tensor([eye], dtype=torch.float32, device=env.device),
            torch.tensor([target], dtype=torch.float32, device=env.device))

    with h5py.File(args.h5, "r") as f:
        q_hl = f["observations/qpos_hand_left"][args.frame].astype(np.float32)
        q_hr = f["observations/qpos_hand_right"][args.frame].astype(np.float32)
        real_aria = f[ARIA_DS][args.frame].astype(np.uint8)
        real_oakd = f[OAKD_DS][args.frame].astype(np.uint8)

    idsL, _ = env.scene["left_arm_hand"].find_joints(list(LEFT_ARM_ORDER) + list(LEFT_HAND_ORDER), preserve_order=True)
    idsR, _ = env.scene["right_arm_hand"].find_joints(list(RIGHT_ARM_ORDER) + list(RIGHT_HAND_ORDER), preserve_order=True)

    def hold(qaL, qaR):
        a = torch.zeros((1, 48), dtype=torch.float32, device=env.device)
        a[0, 0:7] = torch.tensor(qaL, device=env.device); a[0, 7:14] = torch.tensor(qaR, device=env.device)
        a[0, 14:31] = torch.tensor(q_hl, device=env.device); a[0, 31:48] = torch.tensor(q_hr, device=env.device)
        return a

    RH = 360
    rows = []
    first = True
    for h in range(H):
        valsL = np.concatenate([ikL[h], q_hl]).astype(np.float32)
        valsR = np.concatenate([ikR[h], q_hr]).astype(np.float32)
        for skey, ids, vals in [("left_arm_hand", idsL, valsL), ("right_arm_hand", idsR, valsR)]:
            art = env.scene[skey]
            pos = torch.tensor(vals, device=env.device).unsqueeze(0)
            art.write_joint_state_to_sim(pos, torch.zeros_like(pos), joint_ids=ids)
            art.set_joint_position_target(pos, joint_ids=ids)
        env.scene.write_data_to_sim()
        a = hold(ikL[h], ikR[h])
        for _ in range(args.warmup if first else args.settle):
            env.step(a)
        first = False

        sim_aria = env.scene["aria_rgb_cam"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        sim_oakd = env.scene["oakd_front_view"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        ov = {n: env.scene[f"ov_{n}"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
              for n, *_ in OVERVIEWS}

        lab = labels[h]
        rtxt = f"L {resid[h,0]:.0f}mm/{resid[h,1]:.0f}d  R {resid[h,2]:.0f}mm/{resid[h,3]:.0f}d"
        reach = "REACHABLE" if (resid[h, 0] < 30 and resid[h, 2] < 30 and resid[h, 1] < 3 and resid[h, 3] < 3) else "unreach"
        col = (120, 255, 120) if reach == "REACHABLE" else (120, 160, 255)
        row = np.concatenate([
            label(fit(ov["iso"], 480, RH), f"{lab}  [{reach}]", col, 0.6),
            label(fit(ov["top"], 360, RH), rtxt, (255, 230, 120), 0.5),
            label(fit(sim_aria, 480, RH), "SIM aria", (120, 200, 255), 0.6),
            label(fit(real_aria, 480, RH), "REAL aria", (120, 255, 120), 0.6),
            label(fit(sim_oakd, 560, RH), "SIM oakd", (120, 200, 255), 0.6),
            label(fit(real_oakd, 560, RH), "REAL oakd", (120, 255, 120), 0.6),
        ], axis=1)
        rows.append(row)
        # also save a per-candidate full-res iso for detail
        cv2.imwrite(os.path.join(args.out_dir, f"{model}_{lab}_iso.png"),
                    cv2.cvtColor(label(ov["iso"], f"{model} {lab} {rtxt}", col), cv2.COLOR_RGB2BGR))
        log(f"  [{h+1}/{H}] {lab}  {rtxt}  {reach}")

    sheet = np.concatenate(rows, axis=0)
    outp = os.path.join(args.out_dir, f"{model}_orientation_sheet_t{args.frame}.png")
    cv2.imwrite(outp, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    log("wrote", outp)
    log("=" * 60); log("DONE"); env.close(); simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("\n*** EXCEPTION ***"); log(traceback.format_exc()); os._exit(1)
