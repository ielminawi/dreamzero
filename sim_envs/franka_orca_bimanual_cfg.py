"""Configuration for dual Franka + Orca hand bimanual environment.

Camera placement derived from configs/franka_orca_calibration.json.
Action/state space: 48 dims = 2x(7 arm + 17 hand).

Each arm+hand is a single 24-DOF articulation loaded from a combined URDF:
  panda_joint1..7 (7 DOF arm) + 17 DOF Orca hand

ORCA hand joint ordering (17 per hand):
  [0]  wrist
  [1]  thumb_mcp       [5]  index_abd       [8]  middle_abd     [11] ring_abd      [14] pinky_abd
  [2]  thumb_abd       [6]  index_mcp       [9]  middle_mcp     [12] ring_mcp      [15] pinky_mcp
  [3]  thumb_pip       [7]  index_pip       [10] middle_pip     [13] ring_pip      [16] pinky_pip
  [4]  thumb_dip

Generated URDFs: sim_envs/assets/franka_orca_{left,right}.urdf
  (run sim_envs/assets/generate_combined_urdf.py to regenerate)
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

# Arm separation from calibration: ~61cm apart on Y-axis
ARM_SEPARATION_Y = 0.6127

# Paths to combined URDFs
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
LEFT_URDF = os.path.join(_ASSETS_DIR, "franka_orca_left.urdf")
RIGHT_URDF = os.path.join(_ASSETS_DIR, "franka_orca_right.urdf")

# Joint name expressions for actuator config
FRANKA_ARM_JOINTS = "panda_joint[1-7]"
# All 17 hand joints (wrist + 4 fingers x {abd, mcp, pip} + thumb dip)
LEFT_HAND_JOINTS = "left_*"
RIGHT_HAND_JOINTS = "right_*"


@configclass
class FrankaOrcaSceneCfg(InteractiveSceneCfg):
    """Scene with two Franka+Orca arm-hand units, table, and cameras.

    Each arm+hand is a single articulation (24 DOF):
      joints[0:7]  = panda_joint1..7 (arm)
      joints[7:24] = {side}_{wrist, thumb_mcp, ..., pinky_pip} (hand)
    """

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="/World/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="nucleus://Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
    )

    # ---- Left Franka + Orca hand (24 DOF) ----
    left_arm_hand = ArticulationCfg(
        prim_path="/World/LeftArmHand",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=LEFT_URDF,
            fix_base=True,
            merge_fixed_joints=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -ARM_SEPARATION_Y / 2, 0.0),
            joint_pos={
                # Arm joints — default home position
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                # Hand joints — all zero (open hand)
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[FRANKA_ARM_JOINTS],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[LEFT_HAND_JOINTS],
                effort_limit=1.0,
                velocity_limit=5.0,
                stiffness=2.0,
                damping=0.1,
            ),
        },
    )

    # ---- Right Franka + Orca hand (24 DOF) ----
    right_arm_hand = ArticulationCfg(
        prim_path="/World/RightArmHand",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=RIGHT_URDF,
            fix_base=True,
            merge_fixed_joints=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, ARM_SEPARATION_Y / 2, 0.0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[FRANKA_ARM_JOINTS],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[RIGHT_HAND_JOINTS],
                effort_limit=1.0,
                velocity_limit=5.0,
                stiffness=2.0,
                damping=0.1,
            ),
        },
    )

    # ---- Camera 1: "aria_rgb_cam" ----
    # Positioned per left_cam extrinsics from calibration.
    # Training resolution: 480x640
    aria_rgb_cam = CameraCfg(
        prim_path="/World/AriaCam",
        update_period=0.02,  # 50 Hz to match training FPS
        height=480,
        width=640,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
        init_state=CameraCfg.InitialStateCfg(
            # From left_cam extrinsics (cam_to_base): translation = [0.204, -0.255, 0.434]
            pos=(0.204, -0.255, 0.434),
            rot=(0.68, -0.19, 0.68, -0.19),  # Approximate — tune visually in sim
        ),
    )

    # ---- Camera 2: "oakd_front_view" ----
    # Positioned per right_cam extrinsics from calibration.
    # Training resolution: 540x960
    oakd_front_view = CameraCfg(
        prim_path="/World/OakDCam",
        update_period=0.02,  # 50 Hz
        height=540,
        width=960,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
        init_state=CameraCfg.InitialStateCfg(
            # From right_cam extrinsics: translation = [0.175, 0.346, 0.469]
            pos=(0.175, 0.346, 0.469),
            rot=(0.66, -0.22, 0.66, -0.22),  # Approximate — tune visually in sim
        ),
    )


# ---------------------------------------------------------------------------
# Observation terms
# ---------------------------------------------------------------------------

def get_left_arm_joint_pos(env) -> "torch.Tensor":
    """Left arm 7-DoF joint positions (joints 0:7 of left_arm_hand)."""
    return env.scene["left_arm_hand"].data.joint_pos[:, :7]


def get_right_arm_joint_pos(env) -> "torch.Tensor":
    """Right arm 7-DoF joint positions (joints 0:7 of right_arm_hand)."""
    return env.scene["right_arm_hand"].data.joint_pos[:, :7]


def get_left_hand_joint_pos(env) -> "torch.Tensor":
    """Left Orca hand 17-DoF joint positions (joints 7:24 of left_arm_hand)."""
    return env.scene["left_arm_hand"].data.joint_pos[:, 7:24]


def get_right_hand_joint_pos(env) -> "torch.Tensor":
    """Right Orca hand 17-DoF joint positions (joints 7:24 of right_arm_hand)."""
    return env.scene["right_arm_hand"].data.joint_pos[:, 7:24]


def get_aria_rgb(env) -> "torch.Tensor":
    """Aria RGB camera image."""
    return env.scene["aria_rgb_cam"].data.output["rgb"][..., :3]


def get_oakd_front(env) -> "torch.Tensor":
    """OAK-D front view camera image."""
    return env.scene["oakd_front_view"].data.output["rgb"][..., :3]


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

@configclass
class FrankaOrcaBimanualEnvCfg(ManagerBasedEnvCfg):
    """Environment config for dual Franka + Orca bimanual manipulation."""

    # Scene
    scene: FrankaOrcaSceneCfg = FrankaOrcaSceneCfg(num_envs=1, env_spacing=2.5)

    # Simulation
    sim = sim_utils.SimulationCfg(
        dt=0.02,  # 50 Hz physics to match training FPS
        render_interval=1,
    )

    # Max episode length
    episode_length_s = 30.0  # 30 seconds = 1500 steps at 50Hz

    # Observations
    observations = ObservationGroupCfg(
        policy=ObservationGroupCfg(
            concatenate_terms=False,
            terms={
                "left_arm_joint_pos": ObservationTermCfg(func=get_left_arm_joint_pos),
                "right_arm_joint_pos": ObservationTermCfg(func=get_right_arm_joint_pos),
                "left_hand_joint_pos": ObservationTermCfg(func=get_left_hand_joint_pos),
                "right_hand_joint_pos": ObservationTermCfg(func=get_right_hand_joint_pos),
                "aria_rgb_cam": ObservationTermCfg(func=get_aria_rgb),
                "oakd_front_view": ObservationTermCfg(func=get_oakd_front),
            },
        ),
    )
