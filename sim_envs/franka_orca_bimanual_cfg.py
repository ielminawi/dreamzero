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

import numpy as np

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import ActionTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

def _look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """Compute (w, x, y, z) quaternion for an OpenGL camera at *eye* looking at *target*."""
    eye, target, up = np.asarray(eye, np.float64), np.asarray(target, np.float64), np.asarray(up, np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    z = -fwd
    if abs(np.dot(up, z)) > 0.999:
        up = np.array([-1.0, 0.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
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


# Camera positions from calibration extrinsics (cam→base).
# Calibration gives camera pos relative to arm base; Y axis is flipped in calib.
# Camera Y = 0 (centered between arms), Z ≈ 0.45 from calibration.
_ARIA_EYE = (-0.25, 0.0, 0.45)
_OAKD_EYE = (-0.25, 0.0, 0.45)
_WORKSPACE_TARGET = (0.35, 0.0, 0.15)

# Arm separation from calibration: ~61cm apart on Y-axis
ARM_SEPARATION_Y = 0.6127

# Paths to combined URDFs
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
LEFT_URDF = os.path.join(_ASSETS_DIR, "franka_orca_left.urdf")
RIGHT_URDF = os.path.join(_ASSETS_DIR, "franka_orca_right.urdf")

# Joint name expressions for actuator config
FRANKA_ARM_JOINTS = "panda_joint[1-7]"
# All 17 hand joints (wrist + 4 fingers x {abd, mcp, pip} + thumb dip)
LEFT_HAND_JOINTS = "left_.*"
RIGHT_HAND_JOINTS = "right_.*"


@configclass
class FrankaOrcaSceneCfg(InteractiveSceneCfg):
    """Scene with two Franka+Orca arm-hand units, table, and cameras.

    Each arm+hand is a single articulation (24 DOF):
      joints[0:7]  = panda_joint1..7 (arm)
      joints[7:24] = {side}_{wrist, thumb_mcp, ..., pinky_pip} (hand)
    """

    # Dome light for ambient illumination
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 1.0),
            intensity=300.0,
        ),
    )

    # Distant light for directional illumination and shadows
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 1.0, 0.95),
            intensity=400.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.866, 0.0, 0.5, 0.0),  # ~60 deg elevation
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Table (simple cuboid stand-in; swap for USD mesh once Nucleus is available)
    # Top surface at z=0.025 so it sits flush on the ground plane
    table = AssetBaseCfg(
        prim_path="/World/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.8, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2), roughness=1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.025)),
    )

    # ---- Left Franka + Orca hand (24 DOF) ----
    left_arm_hand = ArticulationCfg(
        prim_path="/World/LeftArmHand",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=LEFT_URDF,
            fix_base=True,
            merge_fixed_joints=False,
            joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
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
            joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
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
    # Position from left_cam calibration extrinsics (cam→base).
    # pos = left_arm_base + [tx, -ty, tz]  (Y negated: calib Y is flipped vs sim).
    # Orientation: look-at toward workspace center.
    # Training resolution: 480x640
    aria_rgb_cam = CameraCfg(
        prim_path="/World/Cameras/AriaCam",
        update_period=0.02,  # 50 Hz to match training FPS
        height=480,
        width=640,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0,
            horizontal_aperture=20.955,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_ARIA_EYE,
            rot=_look_at_quat(_ARIA_EYE, _WORKSPACE_TARGET),
            convention="opengl",
        ),
    )

    # ---- Camera 2: "oakd_front_view" ----
    # Position from right_cam calibration extrinsics (cam→base).
    # pos = right_arm_base + [tx, -ty, tz]  (Y negated: calib Y is flipped vs sim).
    # Orientation: look-at toward workspace center.
    # Training resolution: 540x960
    oakd_front_view = CameraCfg(
        prim_path="/World/Cameras/OakDCam",
        update_period=0.02,  # 50 Hz
        height=540,
        width=960,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0,
            horizontal_aperture=20.955,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_OAKD_EYE,
            rot=_look_at_quat(_OAKD_EYE, _WORKSPACE_TARGET),
            convention="opengl",
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

    # 1 physics step per control step (50 Hz control = 50 Hz physics)
    decimation = 1

    # Max episode length
    episode_length_s = 30.0  # 30 seconds = 1500 steps at 50Hz

    # Actions — joint position targets for both arm+hand articulations
    @configclass
    class ActionsCfg:
        left_arm = mdp.JointPositionActionCfg(
            asset_name="left_arm_hand",
            joint_names=[FRANKA_ARM_JOINTS],
        )
        left_hand = mdp.JointPositionActionCfg(
            asset_name="left_arm_hand",
            joint_names=[LEFT_HAND_JOINTS],
        )
        right_arm = mdp.JointPositionActionCfg(
            asset_name="right_arm_hand",
            joint_names=[FRANKA_ARM_JOINTS],
        )
        right_hand = mdp.JointPositionActionCfg(
            asset_name="right_arm_hand",
            joint_names=[RIGHT_HAND_JOINTS],
        )

    actions: ActionsCfg = ActionsCfg()

    # Observations
    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObservationGroupCfg):
            concatenate_terms = False
            left_arm_joint_pos = ObservationTermCfg(func=get_left_arm_joint_pos)
            right_arm_joint_pos = ObservationTermCfg(func=get_right_arm_joint_pos)
            left_hand_joint_pos = ObservationTermCfg(func=get_left_hand_joint_pos)
            right_hand_joint_pos = ObservationTermCfg(func=get_right_hand_joint_pos)
            aria_rgb_cam = ObservationTermCfg(func=get_aria_rgb)
            oakd_front_view = ObservationTermCfg(func=get_oakd_front)

        policy: PolicyCfg = PolicyCfg()

    observations: ObservationsCfg = ObservationsCfg()
