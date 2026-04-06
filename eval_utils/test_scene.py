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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now safe to import Isaac/sim_envs
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

from sim_envs.franka_orca_bimanual_cfg import FrankaOrcaBimanualEnvCfg
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv


def main():
    print("\n=== Creating FrankaOrcaBimanual-v0 environment ===")
    cfg = FrankaOrcaBimanualEnvCfg()
    env = FrankaOrcaBimanualEnv(cfg)

    # Double reset (first loads assets, second stabilises materials)
    print("Reset 1/2 (loading assets)...")
    obs, _ = env.reset()
    print("Reset 2/2 (stabilising)...")
    obs, _ = env.reset()

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
