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
import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import ActionTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

# Camera mounting — taken DIRECTLY from configs/franka_orca_calibration.json extrinsics
# (cam->arm_base), NOT a look-at heuristic.
#   pos = arm_base + [tx, -ty, tz]   (calib Y is flipped vs the sim base frame)
#   rot = the calibration rotation converted to OpenGL (w, x, y, z); this faithfully
#         reproduces the real camera's optical axis AND roll.
# Verified: these reproduce the calibration optical axis to 3 decimals — aria looks
# 53.7deg below horizontal, oak-d 56.0deg (both in front of and above the workspace).
#
# WARNING: do NOT replace these with a _look_at_quat(eye, workspace_target) heuristic.
# Commit 57bc624 "camera" did exactly that and it (a) drifted the eye BEHIND the arm
# bases (aria X +0.204 -> -0.250), (b) flattened the aria pitch to ~27deg, and (c)
# discarded the calibrated roll — so neither sim camera matched the real recording.
_ARIA_EYE = (0.204, -0.051, 0.434)            # left_cam extrinsics -> aria_rgb_cam
_ARIA_ROT = (0.660, 0.230, -0.210, -0.683)    # (w,x,y,z) OpenGL, from left_cam rotation
_OAKD_EYE = (0.175, -0.040, 0.469)            # right_cam extrinsics -> oakd_front_view
_OAKD_ROT = (0.678, 0.235, -0.175, -0.674)    # (w,x,y,z) OpenGL, from right_cam rotation

# Arm separation on Y. The calibration file says 0.6127, but that is an UNDERESTIMATE
# (user determination 2026-06-12: the arms render too close at 0.6-0.7; settled on 0.80).
# Do not blindly trust
# configs/franka_orca_calibration.json's arm_separation_meters / left_base_to_right_base.
ARM_SEPARATION_Y = 0.80

# Base placement: the h5 "left" arm stands at +Y, the "right" arm at -Y (the sim world
# X axis points forward into the workspace, so +Y is image-left for the front cameras...
# but the recorded data's left/right labels are the OPPOSITE side). Verified empirically
# 2026-06-09 with the base-swap diagnostic replay (left/right base positions exchanged,
# commands NOT exchanged, cameras unchanged): only this arrangement reproduces the real
# aria/oak-d videos. Do NOT "fix" this back to left=-Y by symmetry intuition.
_LEFT_BASE_POS = (0.0, +ARM_SEPARATION_Y / 2, 0.0)
_RIGHT_BASE_POS = (0.0, -ARM_SEPARATION_Y / 2, 0.0)

# Table top height (table cuboid is 0.05 thick, centered at z=0.025 -> top at 0.05).
# Module-level (NOT a scene-class attribute: InteractiveSceneCfg treats every class
# attribute as a scene asset and would reject a bare float).
_TABLE_TOP = 0.05

# High-friction material so the Orca fingers can actually GRIP objects. With no material,
# PhysX uses a low default and grasped objects slip out. Applied to the objects + table, and
# as the sim default material (so the URDF-imported finger collision shapes get it too).
_GRIP_MAT = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.5, dynamic_friction=1.2, restitution=0.0,
    # "max" combine => the contact friction is the higher of the two materials, so the
    # object's high friction dominates even though the URDF-imported fingers keep the
    # default material. This is what makes the grip hold without slipping.
    friction_combine_mode="max",
)

# Paths to combined URDFs
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
LEFT_URDF = os.path.join(_ASSETS_DIR, "franka_orca_left.urdf")
RIGHT_URDF = os.path.join(_ASSETS_DIR, "franka_orca_right.urdf")

# ---------------------------------------------------------------------------
# Arm joint-convention remap (sim Franka URDF  <->  training/real data convention)
#
# The recorded teleop/training qpos use a different Franka convention than the standard
# sim URDF (raw j4 ~ +0.95 is outside any real Panda's range). The mapping is a PER-ARM,
# PER-JOINT affine map (the two arms are recorded in DIFFERENT conventions — a shared
# mapping provably cannot fit both):
#     sim_i = sign_i * real_i + offset_i        (per joint, training order j1..j7)
#
# Established 2026-06-10 by systematic search (eval_utils/search_arm_convention.py:
# brute force over per-joint sign + pi/4-grid offsets, scored against hand-landmark
# pixels labelled in the REAL h5 frames + camera-free physical gates), confirmed by
# Isaac teleport-renders at 10 pose-diverse frames (eval_utils/multi_hypothesis_frame0.py,
# output/hyp_final_20260610): elbows-up / forearm-plunging / hand-down configuration
# matches the real video on both arms (e.g. the right forearm crossing close past the
# oak-d camera at t=1809, the descending right wrist at t=3417).
#
# The PREVIOUS shared remap [-pi/4,0,0,-pi,0,*,0] is disproven: it folds both arms up
# high (flange z~0.5) so the hands never even enter either camera view, while the real
# hands work on the table the whole episode (output/snapshot_10frames_20260610).
#
# Known residual (~5-10 cm at the flange, structurally correct): the calibration's
# right-base offset (+5.5 cm z, -6 cm x vs left, configs/franka_orca_calibration.json)
# is not yet modelled, the aria camera is a fisheye approximated by a wide pinhole, and
# the true offsets may sit slightly off the pi/4 grid. Refine against renders, not FK.
#
# Use real_arm_to_sim() before applying a real-convention action to the sim, and
# sim_arm_to_real() before sending sim proprioception to the policy.
import math as _math
_PI = _math.pi
ARM_SIM_FROM_REAL = {
    # sign (+1/-1) and offset (rad) per joint j1..j7
    "left": {
        "sign":   [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
        "offset": [_PI / 4, _PI / 4, -_PI / 4, -_PI / 4, -3 * _PI / 4, _PI, 0.0],
    },
    "right": {
        "sign":   [1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0],
        "offset": [-_PI / 4, _PI / 2, _PI / 4, -_PI / 4, _PI / 4, 0.0, 0.0],
    },
}


def real_arm_to_sim(arm7, side="left"):
    """Map a 7-DOF arm vector from training/real convention to sim URDF convention."""
    import numpy as _np
    c = ARM_SIM_FROM_REAL[side]
    return _np.asarray(arm7, dtype=float) * _np.asarray(c["sign"]) + _np.asarray(c["offset"])


def sim_arm_to_real(arm7, side="left"):
    """Map a 7-DOF arm vector from sim URDF convention to training/real convention."""
    import numpy as _np
    c = ARM_SIM_FROM_REAL[side]
    # inverse of x -> s*x + o with s in {+1,-1}:  x -> s*(x - o)
    return (_np.asarray(arm7, dtype=float) - _np.asarray(c["offset"])) * _np.asarray(c["sign"])


# Joint name expressions for actuator config (regex — order irrelevant, only assigns gains)
FRANKA_ARM_JOINTS = "panda_joint[1-7]"
# All 17 hand joints (wrist + 4 fingers x {abd, mcp, pip} + thumb dip)
LEFT_HAND_JOINTS = "left_.*"
RIGHT_HAND_JOINTS = "right_.*"

# Canonical joint ORDER used by training / the h5 dataset / the URDF (see the
# ORCA hand index map in the module docstring). IsaacLab/PhysX REORDERS the branched
# hand joints on URDF import (groups them BFS-by-depth, e.g. all *_abd before *_mcp),
# so data.joint_pos columns are NOT in this order. We therefore gather joints BY NAME
# in this canonical order for both action terms (JointPositionActionCfg + preserve_order)
# and observations, so the sim state/action layout matches what the policy was trained on.
def _arm_order(side):  # side unused; both arms are panda_joint1..7
    return [f"panda_joint{i}" for i in range(1, 8)]


def _hand_order(side):
    return [
        f"{side}_wrist",
        f"{side}_thumb_mcp", f"{side}_thumb_abd", f"{side}_thumb_pip", f"{side}_thumb_dip",
        f"{side}_index_abd", f"{side}_index_mcp", f"{side}_index_pip",
        f"{side}_middle_abd", f"{side}_middle_mcp", f"{side}_middle_pip",
        f"{side}_ring_abd", f"{side}_ring_mcp", f"{side}_ring_pip",
        f"{side}_pinky_abd", f"{side}_pinky_mcp", f"{side}_pinky_pip",
    ]


LEFT_ARM_ORDER = _arm_order("left")
RIGHT_ARM_ORDER = _arm_order("right")
LEFT_HAND_ORDER = _hand_order("left")
RIGHT_HAND_ORDER = _hand_order("right")


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

    # Ground plane (physics collision only; its grid look is hidden by the floor below)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # ---- Room backdrop (reduce the visual domain gap) ----
    # The policy is a video world-model trained on real RGB of a real room; the default
    # Isaac black-grid void is strongly out-of-distribution. These are VISUAL-ONLY (no
    # collision_props) so they can't interpenetrate the arm bases / hang PhysX; the
    # GroundPlaneCfg above still provides the physics floor. Neutral indoor colors.
    floor_visual = AssetBaseCfg(
        prim_path="/World/Backdrop/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(8.0, 8.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.52, 0.47, 0.42), roughness=0.95),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, -0.011)),
    )
    back_wall = AssetBaseCfg(
        prim_path="/World/Backdrop/BackWall",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 6.0, 2.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.70, 0.67, 0.61), roughness=0.95),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.5, 0.0, 1.2)),
    )

    # Table (simple cuboid stand-in; swap for USD mesh once Nucleus is available).
    # Spans z in [0, 0.05] (top at _TABLE_TOP=0.05). MUST have collision_props, else it
    # is a visual-only ghost and the objects/hands fall straight through it onto the
    # ground plane (looked like "bodies sinking into the table").
    # Shifted forward (back edge at x=0.15) so the solid table does NOT overlap the arm
    # bases at x=0: a Franka base sphere (r=0.06 at z=0) spans z in [-0.06, 0.06] and would
    # interpenetrate a collidable table top at z=0.05, which makes PhysX hang. The arms now
    # reach forward+down onto the table. Objects sit at x in [0.30, 0.38] (well on the table).
    table = AssetBaseCfg(
        prim_path="/World/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 0.8, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_GRIP_MAT,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2), roughness=1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.0, 0.025)),
    )

    # ---- Manipulable objects, approximating the bag_groceries training scene ----
    # The training videos show a white paper grocery bag plus a few small items on the
    # table. No object poses are recorded in the h5, so these are primitive rigid-body
    # stand-ins (visual + physics) placed around the camera workspace target (x~0.35,
    # y in [-0.2,0.2]) so both cameras see them. Table top is at z=_TABLE_TOP
    # (module-level constant; a scene-class attribute must be an asset cfg, not a float).

    grocery_bag = RigidObjectCfg(
        prim_path="/World/Objects/GroceryBag",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.13, 0.19),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_GRIP_MAT,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.90, 0.86), roughness=0.9),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.32, 0.12, _TABLE_TOP + 0.095)),
    )

    white_container = RigidObjectCfg(
        prim_path="/World/Objects/WhiteContainer",
        spawn=sim_utils.CuboidCfg(
            size=(0.085, 0.085, 0.045),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_GRIP_MAT,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, -0.02, _TABLE_TOP + 0.0225)),
    )

    blue_tube = RigidObjectCfg(
        prim_path="/World/Objects/BlueTube",
        spawn=sim_utils.CylinderCfg(
            radius=0.018,
            height=0.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_GRIP_MAT,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.85), roughness=0.5),
        ),
        # lay the cylinder on its side (rotate 90deg about X so its axis is along Y)
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.30, -0.08, _TABLE_TOP + 0.018), rot=(0.7071, 0.7071, 0.0, 0.0)
        ),
    )

    round_item = RigidObjectCfg(
        prim_path="/World/Objects/RoundItem",
        spawn=sim_utils.SphereCfg(
            radius=0.026,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_GRIP_MAT,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.65, 0.20), roughness=0.7),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.38, -0.11, _TABLE_TOP + 0.026)),
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
            pos=_LEFT_BASE_POS,
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
                # Sim-fidelity: stiffness=2/effort=1 was too weak to track commanded
                # grasps (fingers drooped under gravity, lagged the policy). Bumped so
                # the Orca fingers follow position targets (works now that the fingertip
                # links have valid inertia — see generate_combined_urdf.sanitize_inertials).
                effort_limit=5.0,
                velocity_limit=5.0,
                stiffness=20.0,
                damping=1.0,
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
            pos=_RIGHT_BASE_POS,
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
                # Sim-fidelity: see left-hand note above (track grasps, valid inertia).
                effort_limit=5.0,
                velocity_limit=5.0,
                stiffness=20.0,
                damping=1.0,
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
            # aria_rgb_cam is an Aria glasses (egocentric) RGB camera — a WIDE FOV (~100-110 deg),
            # unlike the narrower oak-d. focal_length=8 -> HFOV ~105 deg so the whole table + both
            # hands fill the egocentric view like the training frame (output/bg_aria_ref.png).
            # (The shared focal=20 was tuned for oak-d and is far too narrow for Aria.)
            focal_length=8.0,
            horizontal_aperture=20.955,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_ARIA_EYE,
            rot=_ARIA_ROT,
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
            # focal_length raised 15->20 (HFOV ~70deg -> ~55deg) to tighten the framing
            # toward the real cameras: crops the Isaac floor grid + Franka base links at
            # the edges and enlarges the workspace/objects, while still keeping BOTH arms
            # in view. Tuned by matching a rendered frame to the training video
            # (output/sim_vs_train_oakd*.png). (focal=24 over-zoomed to a single hand.)
            focal_length=20.0,
            horizontal_aperture=20.955,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_OAKD_EYE,
            rot=_OAKD_ROT,
            convention="opengl",
        ),
    )


# ---------------------------------------------------------------------------
# Observation terms
# ---------------------------------------------------------------------------

def _joint_cols(env, scene_key, names):
    """Column indices into data.joint_pos that select `names` in the given (training)
    order. Cached per (scene_key, names). Necessary because Isaac reorders the hand
    joints vs the URDF, so raw slicing [:, 7:24] returns a scrambled order."""
    cache = env.__dict__.setdefault("_dz_joint_cols", {})
    key = (scene_key, tuple(names))
    if key not in cache:
        ids, resolved = env.scene[scene_key].find_joints(list(names), preserve_order=True)
        assert list(resolved) == list(names), (
            f"{scene_key}: joint resolve order mismatch:\n  got={list(resolved)}\n  want={list(names)}"
        )
        cache[key] = ids
    return cache[key]


def get_left_arm_joint_pos(env) -> "torch.Tensor":
    """Left arm 7-DoF joint positions in canonical training order (panda_joint1..7)."""
    return env.scene["left_arm_hand"].data.joint_pos[:, _joint_cols(env, "left_arm_hand", LEFT_ARM_ORDER)]


def get_right_arm_joint_pos(env) -> "torch.Tensor":
    """Right arm 7-DoF joint positions in canonical training order (panda_joint1..7)."""
    return env.scene["right_arm_hand"].data.joint_pos[:, _joint_cols(env, "right_arm_hand", RIGHT_ARM_ORDER)]


def get_left_hand_joint_pos(env) -> "torch.Tensor":
    """Left Orca hand 17-DoF joint positions gathered by name in canonical training order."""
    return env.scene["left_arm_hand"].data.joint_pos[:, _joint_cols(env, "left_arm_hand", LEFT_HAND_ORDER)]


def get_right_hand_joint_pos(env) -> "torch.Tensor":
    """Right Orca hand 17-DoF joint positions gathered by name in canonical training order."""
    return env.scene["right_arm_hand"].data.joint_pos[:, _joint_cols(env, "right_arm_hand", RIGHT_HAND_ORDER)]


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

    # Actions — joint position targets for both arm+hand articulations.
    #
    # The ActionManager concatenates terms in DEFINITION ORDER, so the term order below
    # MUST equal the server/training action layout:
    #     [left_arm(7), right_arm(7), left_hand(17), right_hand(17)]  (indices 0:7,7:14,14:31,31:48)
    #
    # Each term passes an EXPLICIT joint_names list in canonical training order plus
    # preserve_order=True, so action[i] drives joint_names[i] regardless of Isaac's
    # internal (BFS-reordered) joint indexing.
    #
    # use_default_offset=False / scale=1.0: the policy server already returns ABSOLUTE
    # joint targets (GrootSimPolicy.unapply adds the observation state back to the model's
    # relative delta because the checkpoint has relative_action=true). So we apply the
    # action directly as the joint-position target — NO default-pose offset, NO adding the
    # current joint position (the default use_default_offset=True would add the home pose).
    @configclass
    class ActionsCfg:
        left_arm = mdp.JointPositionActionCfg(
            asset_name="left_arm_hand",
            joint_names=LEFT_ARM_ORDER,
            preserve_order=True,
            use_default_offset=False,
            scale=1.0,
        )
        right_arm = mdp.JointPositionActionCfg(
            asset_name="right_arm_hand",
            joint_names=RIGHT_ARM_ORDER,
            preserve_order=True,
            use_default_offset=False,
            scale=1.0,
        )
        left_hand = mdp.JointPositionActionCfg(
            asset_name="left_arm_hand",
            joint_names=LEFT_HAND_ORDER,
            preserve_order=True,
            use_default_offset=False,
            scale=1.0,
        )
        right_hand = mdp.JointPositionActionCfg(
            asset_name="right_arm_hand",
            joint_names=RIGHT_HAND_ORDER,
            preserve_order=True,
            use_default_offset=False,
            scale=1.0,
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
