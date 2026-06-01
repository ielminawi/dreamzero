"""Headless third-person render of the FrankaOrcaBimanual scene for visual debugging.

Spawns a wide third-person camera, resets the env to the home pose, settles a few
render frames, and writes PNGs (third-person + the two policy cameras) to /output.

Camera pose is configurable via env vars so we can iterate without code changes:
    TP_EYE="2.2,0.0,0.8"  TP_TARGET="0.2,0.0,0.35"

Usage (inside the isaac-sim container):
    /opt/IsaacLab/isaaclab.sh -p eval_utils/render_thirdperson.py --headless --enable_cameras
"""

import os
import argparse

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Third-person render of FrankaOrca scene")
parser.add_argument("--out", type=str, default="/output/debug", help="Output dir for PNGs")
parser.add_argument("--settle", type=int, default=24, help="Render/sim steps before capture")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys
sys.path.insert(0, "/app")

# Enable URDF importer extension (same as the eval entrypoint)
import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
for ext_name in ["isaacsim.asset.importer.urdf", "omni.importer.urdf", "omni.isaac.urdf"]:
    if ext_manager.set_extension_enabled_immediate(ext_name, True):
        print(f"[INFO] Enabled extension: {ext_name}")
        break

import imageio.v3 as iio
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
import sim_envs  # noqa: F401
from sim_envs.franka_orca_bimanual_cfg import FrankaOrcaBimanualEnvCfg, _look_at_quat
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv


def _vec(name, default):
    v = os.environ.get(name)
    if not v:
        return default
    return tuple(float(x) for x in v.split(","))


TP_EYE = _vec("TP_EYE", (2.2, 0.0, 0.8))
TP_TARGET = _vec("TP_TARGET", (0.2, 0.0, 0.35))
print(f"[render] third-person eye={TP_EYE} target={TP_TARGET}")

cfg = FrankaOrcaBimanualEnvCfg()

# Several diagnostic viewpoints. Arms sit near x=0, y=+-0.30, base z=0; the table
# slab is in front around x=0.5. These angles let us judge upright-ness + facing.
TARGET = (0.1, 0.0, 0.35)
VIEWS = {
    # reference-style front beauty shot: level, both arms framed, table below
    "reference":   ((2.7, 0.0, 0.45), (0.0, 0.0, 0.45)),
    "front_posX":  ((2.2, 0.0, 0.7),  TARGET),   # looking from +X back toward arms
    "side_posY":   ((0.1, 2.2, 0.7),  TARGET),   # profile from +Y (see arm bend)
    "iso":         ((1.8, 1.8, 1.3),  TARGET),   # three-quarter elevated
}
for name, (eye, tgt) in VIEWS.items():
    setattr(cfg.scene, f"tp_{name}", CameraCfg(
        prim_path=f"/World/Cameras/TP_{name}",
        update_period=0.0, height=720, width=1280,
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(pos=eye, rot=_look_at_quat(eye, tgt), convention="opengl"),
    ))

env = FrankaOrcaBimanualEnv(cfg)
obs, _ = env.reset()

# Settle: step with zero (relative) action so the arms hold the home pose while
# the RTX renderer accumulates a clean frame.
import torch
zero_action = torch.zeros((env.num_envs, 48), device=env.device)
for _ in range(args.settle):
    obs, _ = env.step(zero_action)

os.makedirs(args.out, exist_ok=True)


def _save(cam_key, fname):
    cam = env.scene[cam_key]
    rgb = cam.data.output["rgb"][0, ..., :3].clone().detach().cpu().numpy()
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    path = os.path.join(args.out, fname)
    iio.imwrite(path, rgb)
    print(f"[render] wrote {path}  shape={rgb.shape}")


for name in VIEWS:
    _save(f"tp_{name}", f"tp_{name}.png")
_save("aria_rgb_cam", "aria.png")
_save("oakd_front_view", "oakd.png")

print("[render] done")
simulation_app.close()
