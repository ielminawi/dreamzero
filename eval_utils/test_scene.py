"""Standalone test: load the bimanual Franka+Orca scene in Isaac Sim.

Usage (inside isaac-sim container):
  /opt/IsaacLab/isaaclab.sh -p eval_utils/test_scene.py --headless
  /opt/IsaacLab/isaaclab.sh -p eval_utils/test_scene.py --enable omni.kit.livestream.native

Without --headless, it opens a local viewport.
With livestream, connect via Omniverse Streaming Client on port 8211.
"""

import argparse
import sys
import time

# Isaac Sim must be launched before any other Isaac imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test bimanual Franka+Orca scene loading")
parser.add_argument("--num-steps", type=int, default=500, help="Number of sim steps to run (0 = run forever)")
parser.add_argument("--save-images", action="store_true", help="Save camera renders to /output after reset")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now safe to import Isaac/sim_envs
import numpy as np
import torch
import sys
sys.path.insert(0, "/app")

# Enable URDF importer extension (not loaded by default in headless mode)
import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
for ext_name in ["isaacsim.asset.importer.urdf", "omni.importer.urdf", "omni.isaac.urdf"]:
    if ext_manager.set_extension_enabled_immediate(ext_name, True):
        print(f"[INFO] Enabled extension: {ext_name}")
        break

import sim_envs  # registers FrankaOrcaBimanual-v0

from sim_envs.franka_orca_bimanual_cfg import (
    FrankaOrcaBimanualEnvCfg,
    FrankaOrcaSceneCfg,
)
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass


# ---------------------------------------------------------------------------
# Debug overview cameras (only used with --save-images)
# ---------------------------------------------------------------------------

def _look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """Compute (w, x, y, z) quaternion for an OpenGL camera at *eye* looking at *target*."""
    eye, target, up = np.asarray(eye, np.float64), np.asarray(target, np.float64), np.asarray(up, np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    z = -fwd  # camera looks along -Z
    if abs(np.dot(up, z)) > 0.999:
        up = np.array([-1.0, 0.0, 0.0])  # fallback for top-down
    x = np.cross(up, z);  x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])  # rotation matrix
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, qx, qy, qz = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, qx, qy, qz = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, qx, qy, qz = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, qx, qy, qz = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return (float(w), float(qx), float(qy), float(qz))


# (name, eye, target)
_DEBUG_VIEWS = [
    ("top_down",    (0.3, 0.0, 3.0),  (0.3, 0.0, 0.0)),
    ("front",       (2.0, 0.0, 1.0),  (0.2, 0.0, 0.3)),
    ("left_side",   (0.3, -2.0, 1.0), (0.3, 0.0, 0.3)),
    ("perspective", (1.5, -1.2, 1.5),  (0.3, 0.0, 0.2)),
]

def _make_debug_cam_cfg(prim_name, eye, target):
    return CameraCfg(
        prim_path=f"/World/DebugCams/{prim_name}",
        update_period=0.02,
        height=720,
        width=1280,
        spawn=sim_utils.PinholeCameraCfg(focal_length=15.0, horizontal_aperture=36.0),
        offset=CameraCfg.OffsetCfg(pos=eye, rot=_look_at_quat(eye, target), convention="world"),
    )

@configclass
class DebugSceneCfg(FrankaOrcaSceneCfg):
    """Base scene + debug overview cameras."""
    top_cam: CameraCfg = _make_debug_cam_cfg("TopDown", *_DEBUG_VIEWS[0][1:])
    front_cam: CameraCfg = _make_debug_cam_cfg("Front", *_DEBUG_VIEWS[1][1:])
    left_cam: CameraCfg = _make_debug_cam_cfg("LeftSide", *_DEBUG_VIEWS[2][1:])
    perspective_cam: CameraCfg = _make_debug_cam_cfg("Perspective", *_DEBUG_VIEWS[3][1:])

@configclass
class DebugEnvCfg(FrankaOrcaBimanualEnvCfg):
    """Env config with debug cameras added to the scene."""
    scene: DebugSceneCfg = DebugSceneCfg(num_envs=1, env_spacing=2.5)


_DEBUG_CAM_NAMES = ["top_cam", "front_cam", "left_cam", "perspective_cam"]


def main():
    print("\n=== Creating FrankaOrcaBimanual-v0 environment ===")
    if args.save_images:
        print("  (using DebugEnvCfg with overview cameras)")
        cfg = DebugEnvCfg()
    else:
        cfg = FrankaOrcaBimanualEnvCfg()
    env = FrankaOrcaBimanualEnv(cfg)

    # Double reset (first loads assets, second stabilises materials)
    print("Reset 1/2 (loading assets)...")
    obs, _ = env.reset()
    print("Reset 2/2 (stabilising)...")
    obs, _ = env.reset()

    # Save camera renders if requested
    if args.save_images:
        import cv2
        from pathlib import Path

        # Warm up renderer — first few frames can be incomplete
        print("Warming up renderer (10 steps)...")
        zero_action = torch.zeros(1, 48, device=env.device)
        for _ in range(10):
            obs, _ = env.step(zero_action)

        out_dir = Path("/output")
        out_dir.mkdir(parents=True, exist_ok=True)

        policy = obs.get("policy", obs)
        aria = policy["aria_rgb_cam"][0].cpu().numpy().astype(np.uint8)
        oakd = policy["oakd_front_view"][0].cpu().numpy().astype(np.uint8)

        cv2.imwrite(str(out_dir / "aria_rgb_cam.png"), cv2.cvtColor(aria, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "oakd_front_view.png"), cv2.cvtColor(oakd, cv2.COLOR_RGB2BGR))

        # Side-by-side composite
        aria_resized = cv2.resize(aria, (640, 480))
        oakd_resized = cv2.resize(oakd, (640, 480))
        composite = np.concatenate([aria_resized, oakd_resized], axis=1)
        cv2.imwrite(str(out_dir / "composite.png"), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

        # Debug overview cameras
        print("\nSaving debug overview images...")
        for cam_name, (view_name, _, _) in zip(_DEBUG_CAM_NAMES, _DEBUG_VIEWS):
            rgb = env.scene[cam_name].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            fname = f"debug_{view_name}.png"
            cv2.imwrite(str(out_dir / fname), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"  {fname:<30s}: {rgb.shape}")

        print(f"\n=== Saved camera renders to {out_dir} ===")
        print(f"  aria_rgb_cam.png     : {aria.shape}")
        print(f"  oakd_front_view.png  : {oakd.shape}")
        print(f"  composite.png        : {composite.shape}")

    # Print observation summary
    print("\n=== Observation summary ===")
    for key, val in obs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={tuple(val.shape)}, dtype={val.dtype}")
        else:
            print(f"  {key}: {type(val)}")

    # Print joint positions
    left_arm = obs.get("policy", obs).get("left_arm_joint_pos") if isinstance(obs, dict) and "policy" in obs else obs.get("left_arm_joint_pos")
    right_arm = obs.get("policy", obs).get("right_arm_joint_pos") if isinstance(obs, dict) and "policy" in obs else obs.get("right_arm_joint_pos")
    if left_arm is not None:
        print(f"\n  Left arm joints:  {left_arm.cpu().numpy().flatten()}")
    if right_arm is not None:
        print(f"  Right arm joints: {right_arm.cpu().numpy().flatten()}")

    print(f"\n=== Scene loaded successfully! Running {'forever' if args.num_steps == 0 else f'{args.num_steps} steps'}... ===")
    print("  (Ctrl+C to stop)\n")

    # Step with zero actions
    step = 0
    zero_action = torch.zeros(1, 48, device=env.device)
    try:
        while args.num_steps == 0 or step < args.num_steps:
            obs, _ = env.step(zero_action)
            step += 1
            if step % 100 == 0:
                print(f"  Step {step}...")
    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps.")

    print("\n=== Test complete — scene loads and steps correctly ===")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
