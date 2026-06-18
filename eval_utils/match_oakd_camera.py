"""Sweep candidate oak-d camera placements in ONE Isaac session to match the training view.

Drives the arms to recorded qpos (the GT frames show measured pose), then for each
candidate (eye, view direction, focal length) repositions the oakd camera at runtime
and captures a frame. Compare host-side against the GT h5 frames at the same steps
(output/cam_match/gt_oakd_*.png); the arms are the only objects with known geometry,
so arm overlap is the alignment signal.

Candidates come from a JSON file so iteration does not require editing this script:
  [{"name": "calib", "eye": [x,y,z], "dir": [dx,dy,dz], "focal": 15.25}, ...]
"dir" is the viewing direction (OpenGL -Z); "target" may be given instead.
"""
import argparse
import json
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--h5", type=str,
                    default="/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/20250826_111157.h5")
parser.add_argument("--steps", type=int, nargs="+", default=[50, 1000, 1600, 2400])
parser.add_argument("--cams", type=str, default="/app/output/cam_match/candidates.json")
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
from sim_envs.franka_orca_bimanual_cfg import FrankaOrcaBimanualEnvCfg, _look_at_quat
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv
from eval_utils import ee_ik


def set_cam(cam, eye, direction, focal):
    """Reposition + refocus the camera at runtime; returns the applied focal."""
    eye = np.asarray(eye, np.float64)
    direction = np.asarray(direction, np.float64)
    direction = direction / np.linalg.norm(direction)
    quat = _look_at_quat(eye, eye + direction)
    pos_t = torch.tensor([eye], dtype=torch.float32, device=cam.device)
    quat_t = torch.tensor([quat], dtype=torch.float32, device=cam.device)
    cam.set_world_poses(pos_t, quat_t, convention="opengl")
    prim = cam._sensor_prims[0]
    prim.GetFocalLengthAttr().Set(float(focal))
    return float(focal)


def main():
    with open(args.cams) as f:
        candidates = json.load(f)
    for c in candidates:
        if "dir" not in c:
            t = np.asarray(c["target"], np.float64); e = np.asarray(c["eye"], np.float64)
            c["dir"] = (t - e).tolist()

    # qpos_arm_* are EE POSES [x,y,z,qx,qy,qz,qw] (flange in arm base frame), NOT
    # joint angles — IK them to sim-URDF joints (see ee_ik.py / EE_POSE_AND_IK_PIPELINE.md).
    with h5py.File(args.h5, "r") as f:
        pl = f["observations/qpos_arm_left"][:]; pr = f["observations/qpos_arm_right"][:]
        hl = f["observations/qpos_hand_left"][:]; hr = f["observations/qpos_hand_right"][:]
    T = min(len(pl), len(pr), len(hl), len(hr))

    def step_action(st):
        qL, eL, _ = ee_ik.solve_ik(pl[st], ee_ik.Q_HOME)
        qR, eR, _ = ee_ik.solve_ik(pr[st], ee_ik.Q_HOME)
        print(f"step {st}: IK residual L={eL*1000:.1f}mm R={eR*1000:.1f}mm", flush=True)
        return np.concatenate([qL, qR, hl[st], hr[st]]).astype(np.float32)

    cfg = FrankaOrcaBimanualEnvCfg()
    cfg.sim.device = args.device
    env = FrankaOrcaBimanualEnv(cfg)
    env.reset(); env.reset()
    cam = env.scene["oakd_front_view"]

    for st in args.steps:
        st = min(st, T - 1)
        act = torch.tensor(step_action(st), dtype=torch.float32, device=env.device).unsqueeze(0)
        for _ in range(60):
            obs, _ = env.step(act)
        for c in candidates:
            set_cam(cam, c["eye"], c["dir"], c.get("focal", 15.25))
            for _ in range(3):  # flush the render pipeline at the new pose
                obs, _ = env.step(act)
            # data.intrinsic_matrices is stale after a runtime focal change; report the prim attr
            f_applied = cam._sensor_prims[0].GetFocalLengthAttr().Get()
            frame = obs["policy"]["oakd_front_view"][0, ..., :3].cpu().numpy().astype(np.uint8)
            out = f"{args.out}/sim_{c['name']}_step{st}.png"
            cv2.imwrite(out, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"step {st} cam {c['name']}: saved {out}  focal={f_applied:.2f}", flush=True)
    print("DONE", flush=True)
    env.close(); simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback, os
        print("*** EXCEPTION ***\n" + traceback.format_exc(), flush=True); os._exit(1)
