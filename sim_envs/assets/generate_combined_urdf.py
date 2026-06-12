#!/usr/bin/env python3
"""Generate combined Franka Panda + Orca Hand URDFs for Isaac Sim.

Creates two URDFs:
  - franka_orca_left.urdf  (left Franka arm + left Orca hand)
  - franka_orca_right.urdf (right Franka arm + right Orca hand)

Each URDF has a single articulation chain:
  panda_link0 -> panda_joint1..7 -> panda_link8 -> (fixed) -> orca_hand (17 DOF)

Total: 24 DOF per arm+hand (7 arm + 17 hand)

The Franka arm uses simple cylinder/box primitives for visual/collision geometry
(Isaac Sim will render these fine; for higher quality, swap in Franka meshes or
use the Nucleus USD for the arm portion).

Usage:
    python sim_envs/assets/generate_combined_urdf.py

    # Then in Isaac Sim / IsaacLab, load the generated URDFs:
    #   sim_envs/assets/franka_orca_left.urdf
    #   sim_envs/assets/franka_orca_right.urdf
"""

from __future__ import annotations

import copy
import os
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORCA_URDF_PATH = os.path.join(SCRIPT_DIR, "orcahand_description", "models", "urdf", "orcahand_right.urdf")
# Proper LEFT hand URDF, generated from the upstream xacro with the chirality flag (which
# correctly flips joint AXES + per-joint rpy, not just mesh scale). The old in-script
# mirror_orca_hand_to_left() did NOT flip axes, so the left thumb landed on the wrong side
# and finger joints moved erratically. orcahand_left.urdf is checked in (sim_envs/assets/);
# regenerate it from the upstream xacro with:
#   ~/.venvs/dz_h5/bin/xacro orcahand_description/models/urdf/orcahand.urdf.xacro \
#       prefix:=left_ chirality:=left -o sim_envs/assets/orcahand_left.urdf
ORCA_LEFT_URDF_PATH = os.path.join(SCRIPT_DIR, "orcahand_left.urdf")
OUTPUT_DIR = SCRIPT_DIR

# Extracted from dual_franka_mounted_visual.usda:
#   panda_link8 -> Mounted_Gavin -> Mounted_Orca
#
# The mounted USD uses orca_v1, whose root is at the tower. The generated URDF
# uses orcahand_description, whose root is offset from the tower, so the xyz
# values below are corrected to mount the URDF root while preserving the USD
# tower pose.
# 2026-06-11 re-trim (user-driven, against zero-pose renders): the USD-extracted
# values put the LEFT hand ~5cm outboard of its arm axis (left x=0.0578 vs right
# x=0.0063) and both hands ~8cm behind the flange. New values = (a) left re-centered
# to exactly mirror the right, (b) BOTH hands shifted +4cm world-forward at zero
# pose. Derived numerically from measured zero-pose transforms (mount-local =
# R_gavin^T * desired world offset). Previous USD-extracted values:
#   left  "0.0577817 -0.0810538 0.0690376"   right "0.00630262 0.0702546 0.0842720"
# Left mount (empirically confirmed 2026-06-11, render option "Y"):
#   xyz = right xyz with y negated.
#   rpy = the y-mirror of the right rpy (roll & yaw negated), THEN rotated 180deg about the
#         hand-root LOCAL Y axis. The extra 180deg is required because the left hand is the
#         proper chirality xacro hand (orcahand_left.urdf), whose root frame is reflected
#         about the hand's local x; without it the left hand closed 180deg the wrong way.
#   (right rpy "1.737739 0.047154 -0.089270" -> y-mirror "-1.737739 0.047154 0.089270"
#    -> +180deg about root-local Y -> "1.403854 -0.047154 3.052323".)
# 2026-06-11 left-hand trim (user-confirmed against renders): the left hand sat ~4cm off
# along the mount-local Y; left y = right-mirrored y (-0.03045) - 0.04.
DEFAULT_MOUNT_XYZ = {
    "left": "0.006220897 -0.07045047 0.08067889",
    "right": "0.006220897 0.03045047 0.08067889",
}
DEFAULT_MOUNT_RPY = {
    "left": "1.403854 -0.047154 3.052323",
    "right": "1.737739 0.047154 -0.089270",
}

# Extra yaw of the WHOLE mount+hand assembly about the flange (panda_link8 z) axis.
# NOTE: at the all-zero pose the link8 frame is rotated ~pi about X vs world (the
# joint rpy offsets sum to 180deg), so link8 +z points world-DOWN and a positive
# link8-yaw appears CLOCKWISE in a top-down view. Seen from above these values
# therefore rotate: left 90deg CCW, right 90deg CW, turning the fingers from
# pointing inward (toward the other arm, output/sim/zero_pose/zero_top.png
# pre-fix) to pointing FORWARD (+X, toward the table).
DEFAULT_MOUNT_YAW = {
    "left": "-1.5707963",
    "right": "1.5707963",
}


# ---------------------------------------------------------------------------
# Franka Panda arm definition (standard kinematics + simplified geometry)
# ---------------------------------------------------------------------------
# DH-derived link-to-link transforms (xyz offset of child joint in parent frame)
# and joint axes, from the official Franka Panda URDF.
FRANKA_JOINTS = [
    # (name, parent_link, child_link, xyz, rpy, axis, lower, upper)
    ("panda_joint1", "panda_link0", "panda_link1",
     (0, 0, 0.333), (0, 0, 0), (0, 0, 1), -2.8973, 2.8973),
    ("panda_joint2", "panda_link1", "panda_link2",
     (0, 0, 0), (-1.5708, 0, 0), (0, 0, 1), -1.7628, 1.7628),
    ("panda_joint3", "panda_link2", "panda_link3",
     (0, -0.316, 0), (1.5708, 0, 0), (0, 0, 1), -2.8973, 2.8973),
    ("panda_joint4", "panda_link3", "panda_link4",
     (0.0825, 0, 0), (1.5708, 0, 0), (0, 0, 1), -3.0718, -0.0698),
    ("panda_joint5", "panda_link4", "panda_link5",
     (-0.0825, 0.384, 0), (-1.5708, 0, 0), (0, 0, 1), -2.8973, 2.8973),
    # lower limit widened from the textbook Panda -0.0175 to -1.0: with the 90-deg hand
    # bracket modelled (MOUNT_RPY below), joint6 carries only the small real wrist flexion,
    # and the recorded franka_orca data runs joint6 down to ~-0.55 (non-standard convention).
    ("panda_joint6", "panda_link5", "panda_link6",
     (0, 0, 0), (1.5708, 0, 0), (0, 0, 1), -1.0, 3.7525),
    ("panda_joint7", "panda_link6", "panda_link7",
     (0.088, 0, 0), (1.5708, 0, 0), (0, 0, 1), -2.8973, 2.8973),
]

# panda_link7 -> panda_link8 (flange) fixed joint
FRANKA_FLANGE = ("panda_joint8", "panda_link7", "panda_link8",
                 (0, 0, 0.107), (0, 0, 0))

# Link properties: (name, mass_kg, radius, length, color_rgba)
FRANKA_LINKS = [
    ("panda_link0", 0.0,   0.06, 0.05, "0.9 0.9 0.9 1"),  # base (fixed)
    ("panda_link1", 4.970, 0.06, 0.19, "0.9 0.9 0.9 1"),
    ("panda_link2", 0.646, 0.06, 0.00, "0.9 0.9 0.9 1"),
    ("panda_link3", 3.228, 0.06, 0.19, "0.9 0.9 0.9 1"),
    ("panda_link4", 3.587, 0.06, 0.00, "0.9 0.9 0.9 1"),
    ("panda_link5", 1.225, 0.06, 0.19, "0.9 0.9 0.9 1"),
    ("panda_link6", 1.666, 0.05, 0.00, "0.95 0.95 0.95 1"),
    ("panda_link7", 0.735, 0.04, 0.06, "0.3 0.3 0.3 1"),
    ("panda_link8", 0.0,   0.02, 0.01, "0.3 0.3 0.3 1"),  # flange
]


def _make_inertia_elem(mass: float) -> ET.Element:
    """Create a simplified inertial element for a given mass."""
    inertial = ET.SubElement(ET.Element("dummy"), "inertial")
    origin = ET.SubElement(inertial, "origin")
    origin.set("xyz", "0 0 0")
    origin.set("rpy", "0 0 0")
    m = ET.SubElement(inertial, "mass")
    m.set("value", str(mass))
    # Simple sphere inertia approximation
    i_val = max(mass * 0.01, 1e-6)
    inertia = ET.SubElement(inertial, "inertia")
    inertia.set("ixx", f"{i_val:.6f}")
    inertia.set("ixy", "0")
    inertia.set("ixz", "0")
    inertia.set("iyy", f"{i_val:.6f}")
    inertia.set("iyz", "0")
    inertia.set("izz", f"{i_val:.6f}")
    return inertial


def _parse_xyz(text: str) -> tuple[float, float, float]:
    parts = [float(x) for x in text.split()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 floats, got {text!r}")
    return tuple(parts)


def _add_mount_link(robot: ET.Element, side: str, hand_xyz: str, hand_rpy: str) -> str:
    """Add a fixed Gavin-mount link between the Franka flange and Orca hand.

    The real bracket mesh is available as a binary USD at the repo root, but
    URDF importers are much more reliable with native URDF primitives. These
    plate visuals make the mount explicit in the articulation while the fixed
    joint below carries the authoritative extracted pose.
    """
    link_name = f"{side}_gavin_mount"
    link = ET.SubElement(robot, "link")
    link.set("name", link_name)

    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.05"})
    ET.SubElement(inertial, "inertia", {
        "ixx": "0.000010000",
        "ixy": "0",
        "ixz": "0",
        "iyy": "0.000010000",
        "iyz": "0",
        "izz": "0.000010000",
    })

    def add_plate(name: str, xyz: str, rpy: str, size: str) -> None:
        visual = ET.SubElement(link, "visual")
        visual.set("name", name)
        ET.SubElement(visual, "origin", {"xyz": xyz, "rpy": rpy})
        geom = ET.SubElement(visual, "geometry")
        ET.SubElement(geom, "box", {"size": size})
        ET.SubElement(visual, "material", {"name": "dark"})

    hx, hy, hz = _parse_xyz(hand_xyz)
    add_plate("gavin_flange_plate", "0 0 0.006", "0 0 0", "0.070 0.070 0.012")
    add_plate("gavin_hand_plate", f"{hx:.7g} {hy:.7g} {hz:.7g}", hand_rpy, "0.065 0.055 0.012")
    add_plate("gavin_offset_body", f"{hx * 0.5:.7g} {hy * 0.5:.7g} {hz * 0.5:.7g}", "0 0 0", "0.030 0.030 0.090")

    return link_name


def build_franka_arm_xml() -> ET.Element:
    """Build the Franka Panda arm as an XML element tree (links + joints)."""
    robot = ET.Element("robot")
    robot.set("name", "franka_orca")

    # Material definitions
    for name, rgba in [("white", "1 1 1 1"), ("grey", "0.9 0.9 0.9 1"),
                        ("dark", "0.3 0.3 0.3 1"),
                        ("orca_black", ORCA_BLACK_RGBA),
                        ("orca_skin", ORCA_SKIN_RGBA)]:
        mat = ET.SubElement(robot, "material")
        mat.set("name", name)
        color = ET.SubElement(mat, "color")
        color.set("rgba", rgba)

    # Real Franka Panda meshes (link0..link7) from the official franka_description
    # package, copied out of the Isaac Sim URDF-importer assets into
    # sim_envs/assets/franka_description/meshes/. Each linkN.dae is authored in the
    # panda_linkN frame, so it attaches with an identity visual origin — exactly as
    # in the upstream Franka URDF whose kinematics FRANKA_JOINTS mirrors. link8 is a
    # virtual flange (no mesh); the Orca hand mounts there.
    FRANKA_VISUAL_MESH = {f"panda_link{i}": f"link{i}" for i in range(8)}

    # Create links with the real Franka meshes
    for link_name, mass, radius, length, color_rgba in FRANKA_LINKS:
        link = ET.SubElement(robot, "link")
        link.set("name", link_name)

        # Inertial
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin").set("xyz", "0 0 0")
        ET.SubElement(inertial, "origin").set("rpy", "0 0 0")
        inertial.find("origin").set("rpy", "0 0 0")
        ET.SubElement(inertial, "mass").set("value", str(mass))
        i_val = max(mass * 0.001, 1e-9)
        inertia = ET.SubElement(inertial, "inertia")
        for attr, val in [("ixx", i_val), ("ixy", 0), ("ixz", 0),
                          ("iyy", i_val), ("iyz", 0), ("izz", i_val)]:
            inertia.set(attr, f"{val:.9f}")

        mesh_name = FRANKA_VISUAL_MESH.get(link_name)
        if mesh_name is not None:
            # Visual — real Franka mesh (DAE carries the white Franka materials)
            visual = ET.SubElement(link, "visual")
            ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            vg = ET.SubElement(visual, "geometry")
            ET.SubElement(vg, "mesh", {
                "filename": f"franka_description/meshes/visual/{mesh_name}.dae"})

            # Collision — matching convex STL
            collision = ET.SubElement(link, "collision")
            ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            cg = ET.SubElement(collision, "geometry")
            ET.SubElement(cg, "mesh", {
                "filename": f"franka_description/meshes/collision/{mesh_name}.stl"})

    # Create revolute joints
    for jname, parent, child, xyz, rpy, axis, lower, upper in FRANKA_JOINTS:
        joint = ET.SubElement(robot, "joint")
        joint.set("name", jname)
        joint.set("type", "revolute")
        ET.SubElement(joint, "parent").set("link", parent)
        ET.SubElement(joint, "child").set("link", child)
        origin = ET.SubElement(joint, "origin")
        origin.set("xyz", f"{xyz[0]} {xyz[1]} {xyz[2]}")
        origin.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")
        ax = ET.SubElement(joint, "axis")
        ax.set("xyz", f"{axis[0]} {axis[1]} {axis[2]}")
        limit = ET.SubElement(joint, "limit")
        limit.set("lower", str(lower))
        limit.set("upper", str(upper))
        limit.set("effort", "87.0")
        limit.set("velocity", "2.175")
        dynamics = ET.SubElement(joint, "dynamics")
        dynamics.set("damping", "10.0")
        dynamics.set("friction", "0.1")

    # Flange fixed joint
    jname, parent, child, xyz, rpy = FRANKA_FLANGE
    joint = ET.SubElement(robot, "joint")
    joint.set("name", jname)
    joint.set("type", "fixed")
    ET.SubElement(joint, "parent").set("link", parent)
    ET.SubElement(joint, "child").set("link", child)
    origin = ET.SubElement(joint, "origin")
    origin.set("xyz", f"{xyz[0]} {xyz[1]} {xyz[2]}")
    origin.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")

    return robot


def parse_orca_hand(urdf_path: str) -> ET.Element:
    """Parse the Orca hand URDF and return the root element."""
    tree = ET.parse(urdf_path)
    return tree.getroot()


# Real ORCA v1 look: near-black FDM-printed structure + light-gray cast-silicone
# skin pads. The upstream URDF assigns "white" to every visual, which is wrong for
# the printed parts. (The official MJCF uses 0.25 black / 1.0 white; these were
# tuned darker/grayer against renders 2026-06-12.)
ORCA_BLACK_RGBA = "0.08 0.08 0.08 1"
ORCA_SKIN_RGBA = "0.78 0.78 0.78 1"
_WHITE_MESH_RE = re.compile(r"(_skin_mesh|tower_text_mesh)\.stl$")


def apply_two_tone_materials(orca_robot: ET.Element) -> tuple[int, int]:
    """Assign black to printed-structure visuals, light gray to silicone-skin ones.

    Classification is by visual mesh filename: ``*_skin_mesh.stl`` and the tower
    logo ``tower_text_mesh.stl`` get ``orca_skin``; every other visual (finger/palm
    structure, tower hull/main/fans/camera/u2d2) becomes ``orca_black``. Both
    definitions live with the other root-level materials in build_franka_arm_xml()
    (it cannot be defined here: combine_franka_orca() copies links before
    materials, so by the time its material pass runs, the bare ``orca_black``
    references inside the links already make the name look "existing" and a
    definition appended here would be skipped).
    Returns (n_black, n_white) visual counts.
    """
    n_black = n_white = 0
    for visual in orca_robot.iter("visual"):
        geometry = visual.find("geometry")
        mesh = geometry.find("mesh") if geometry is not None else None
        if mesh is None:
            continue
        basename = os.path.basename(mesh.get("filename", ""))
        material = visual.find("material")
        if material is None:
            material = ET.SubElement(visual, "material")
        if _WHITE_MESH_RE.search(basename):
            material.set("name", "orca_skin")
            n_white += 1
        else:
            material.set("name", "orca_black")
            n_black += 1
    return n_black, n_white


def resolve_mesh_paths(element: ET.Element, orca_description_dir: str,
                        output_dir: str) -> None:
    """Replace package:// mesh paths with paths relative to the output URDF.

    Uses relative paths so the URDF works both locally and inside Docker
    containers (as long as the orcahand_description directory is in the
    same relative location).
    """
    for mesh in element.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith("package://orcahand_description/"):
            rel_path = filename.replace("package://orcahand_description/", "")
            abs_mesh = os.path.join(orca_description_dir, rel_path)
            # Make relative to the output directory
            rel_to_output = os.path.relpath(abs_mesh, output_dir)
            mesh.set("filename", rel_to_output)


def mirror_orca_hand_to_left(orca_root: ET.Element) -> ET.Element:
    """Create a left-hand variant by mirroring the right-hand URDF.

    Mirroring strategy (matches the XACRO chirality logic):
    - Rename all "right_" prefixes to "left_"
    - Negate x-components of joint origins and axes where appropriate
    - Add scale="-1 1 1" to mesh geometry to mirror visuals
    """
    # Deep copy and work on the copy
    left = copy.deepcopy(orca_root)

    # Rename all right_ to left_ in names and references
    for elem in left.iter():
        for attr in ["name", "link", "joint"]:
            val = elem.get(attr, "")
            if "right_" in val:
                elem.set(attr, val.replace("right_", "left_"))
        # Also handle text content if any
        if elem.text and "right_" in elem.text:
            elem.text = elem.text.replace("right_", "left_")

    # Mirror x-axis: negate x in joint origins and some axes
    # Based on the XACRO: s = -1 for left chirality
    for joint in left.iter("joint"):
        origin = joint.find("origin")
        if origin is not None:
            xyz = origin.get("xyz", "0 0 0")
            parts = [float(x) for x in xyz.split()]
            if len(parts) == 3:
                parts[0] = -parts[0]  # Negate x
                origin.set("xyz", f"{parts[0]} {parts[1]} {parts[2]}")

            rpy = origin.get("rpy", "0 0 0")
            parts = [float(x) for x in rpy.split()]
            if len(parts) == 3:
                # Mirror: negate y and z rotations (pitch and yaw)
                parts[1] = -parts[1]
                parts[2] = -parts[2]
                origin.set("rpy", f"{parts[0]} {parts[1]} {parts[2]}")

        # NOTE: do NOT negate the joint axis. The real Orca left hand uses the SAME
        # per-joint rotation convention as the right (recorded teleop/state confirm
        # "flex = positive" on both hands). Finger joints have axes in the Y/Z plane
        # (x=0), so negating x is a no-op for them; the only nonzero-x axis is the wrist,
        # and negating it INVERTED the left wrist (a positive command rotated it the
        # wrong way) with an un-swapped limit range. Keeping the axis as-is matches the
        # recorded data. The link FRAME mirroring (origin x, rpy y/z, mesh scale -1)
        # below still produces the correct mirrored left-hand geometry.

    # Mirror link inertial origins (negate x)
    for link in left.iter("link"):
        for inertial in link.iter("inertial"):
            origin = inertial.find("origin")
            if origin is not None:
                xyz = origin.get("xyz", "0 0 0")
                parts = [float(x) for x in xyz.split()]
                if len(parts) == 3:
                    parts[0] = -parts[0]
                    origin.set("xyz", f"{parts[0]} {parts[1]} {parts[2]}")

    # Mirror visual/collision origins and add mesh scale for mirroring
    for tag in ["visual", "collision"]:
        for elem in left.iter(tag):
            origin = elem.find("origin")
            if origin is not None:
                xyz = origin.get("xyz", "0 0 0")
                parts = [float(x) for x in xyz.split()]
                if len(parts) == 3:
                    parts[0] = -parts[0]
                    origin.set("xyz", f"{parts[0]} {parts[1]} {parts[2]}")

            # Add mirroring scale to meshes
            geometry = elem.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None:
                    mesh.set("scale", "-1 1 1")

    return left


def sanitize_inertials(robot: ET.Element, min_mass: float = 0.01,
                       min_inertia: float = 1e-6) -> list[str]:
    """Give zero-mass leaf links (Orca fingertips + hand root) a small valid
    mass + diagonal inertia. PhysX flags mass=0 rigid bodies as having an invalid
    inertia tensor and substitutes a bad default (negative mass), which makes the
    hand dynamics unstable. The fixed base / flange (panda_link0/8) are left alone
    (they are fixed-jointed, no inertia needed). Returns the names fixed."""
    import re
    target = re.compile(r"(fingertip|_root)$")
    fixed = []
    for link in robot.findall("link"):
        name = link.get("name", "")
        if not target.search(name):
            continue
        inert = link.find("inertial")
        if inert is None:
            continue
        m = inert.find("mass")
        I = inert.find("inertia")
        if m is None or I is None or float(m.get("value", "0")) != 0.0:
            continue
        m.set("value", str(min_mass))
        for a in ("ixx", "iyy", "izz"):
            I.set(a, f"{min_inertia:.9f}")
        for a in ("ixy", "ixz", "iyz"):
            I.set(a, "0")
        fixed.append(name)
    return fixed


def combine_franka_orca(franka_robot: ET.Element, orca_robot: ET.Element,
                         side: str, mount_rpy: str = "0 0 0",
                         mount_xyz: str = "0 0 0",
                         mount_yaw: str = "0") -> ET.Element:
    """Combine Franka arm with Orca hand into a single URDF.

    Adds a fixed joint from panda_link8 to the hand's root link.

    Args:
        franka_robot: Franka arm XML tree
        orca_robot: Orca hand XML tree (already has correct prefix)
        side: "left" or "right"
        mount_rpy: RPY rotation of hand relative to Franka flange
        mount_xyz: XYZ offset of hand relative to Franka flange
        mount_yaw: extra yaw (rad) of the whole mount+hand assembly about the
            flange z axis, applied at the panda_link8 -> gavin_mount joint
    """
    combined = copy.deepcopy(franka_robot)
    combined.set("name", f"franka_orca_{side}")

    hand_root_link = f"{side}_root"

    # Copy all links and joints from the Orca hand
    for link in orca_robot.iter("link"):
        combined.append(copy.deepcopy(link))

    for joint in orca_robot.iter("joint"):
        combined.append(copy.deepcopy(joint))

    # Copy material definitions
    for material in orca_robot.iter("material"):
        # Avoid duplicates
        mat_name = material.get("name")
        existing = [m for m in combined.iter("material") if m.get("name") == mat_name]
        if not existing:
            combined.append(copy.deepcopy(material))

    # Add the Gavin 90-degree mount as an explicit fixed link, then attach the
    # hand root with the transform extracted from the mounted USD scene.
    mount_link = _add_mount_link(combined, side, mount_xyz, mount_rpy)

    flange_joint = ET.SubElement(combined, "joint")
    flange_joint.set("name", f"panda_link8_to_{side}_gavin_mount")
    flange_joint.set("type", "fixed")
    ET.SubElement(flange_joint, "parent").set("link", "panda_link8")
    ET.SubElement(flange_joint, "child").set("link", mount_link)
    ET.SubElement(flange_joint, "origin", {"xyz": "0 0 0", "rpy": f"0 0 {mount_yaw}"})

    mount_joint = ET.SubElement(combined, "joint")
    mount_joint.set("name", f"gavin_mount_to_{side}_hand")
    mount_joint.set("type", "fixed")
    ET.SubElement(mount_joint, "parent").set("link", mount_link)
    ET.SubElement(mount_joint, "child").set("link", hand_root_link)
    origin = ET.SubElement(mount_joint, "origin")
    origin.set("xyz", mount_xyz)
    origin.set("rpy", mount_rpy)

    # Orca fingertip / root markers are massless; give them valid inertia so PhysX
    # doesn't substitute a bad default (see sanitize_inertials).
    fixed = sanitize_inertials(combined)
    if fixed:
        print(f"  [{side}] sanitized zero-mass links: {fixed}")

    return combined


def prettify_xml(elem: ET.Element) -> str:
    """Return pretty-printed XML string."""
    rough = ET.tostring(elem, encoding="unicode")
    parsed = minidom.parseString(rough)
    lines = parsed.toprettyxml(indent="  ").split("\n")
    # Remove extra blank lines and XML declaration
    return "\n".join(line for line in lines
                     if line.strip() and not line.startswith("<?xml"))


def main():
    orca_description_dir = os.path.join(SCRIPT_DIR, "orcahand_description")

    # Parse the right hand URDF
    print(f"Parsing Orca hand URDF: {ORCA_URDF_PATH}")
    orca_right = parse_orca_hand(ORCA_URDF_PATH)

    # Resolve mesh paths to be relative to output directory
    resolve_mesh_paths(orca_right, orca_description_dir, OUTPUT_DIR)
    n_black, n_white = apply_two_tone_materials(orca_right)
    print(f"[materials] right hand: {n_black} black / {n_white} white visuals")

    # LEFT hand: parse the PROPER left URDF (xacro chirality:=left) instead of mirroring
    # the right in-script. The xacro flips joint axes + per-joint rpy correctly; the old
    # mirror_orca_hand_to_left() did not, which put the thumb on the wrong side.
    print(f"Parsing proper LEFT hand URDF: {ORCA_LEFT_URDF_PATH}")
    orca_left = parse_orca_hand(ORCA_LEFT_URDF_PATH)
    resolve_mesh_paths(orca_left, orca_description_dir, OUTPUT_DIR)
    n_black, n_white = apply_two_tone_materials(orca_left)
    print(f"[materials] left hand: {n_black} black / {n_white} white visuals")

    # Build Franka arm
    print("Building Franka Panda arm...")
    franka = build_franka_arm_xml()

    # Mounting transform: how the Orca hand attaches to the Franka flange (panda_link8).
    #
    # The real setup uses the Gavin 90-degree bracket, so the hand hangs roughly
    # perpendicular to the wrist axis instead of straight off the Franka flange.
    # The side-specific defaults come from the baked local matrices in
    # dual_franka_mounted_visual.usda. Env overrides are kept for render sweeps.
    mount_xyz_left = os.environ.get("MOUNT_XYZ_LEFT", os.environ.get("MOUNT_XYZ", DEFAULT_MOUNT_XYZ["left"]))
    mount_rpy_left = os.environ.get("MOUNT_RPY_LEFT", os.environ.get("MOUNT_RPY", DEFAULT_MOUNT_RPY["left"]))
    mount_xyz_right = os.environ.get("MOUNT_XYZ_RIGHT", os.environ.get("MOUNT_XYZ", DEFAULT_MOUNT_XYZ["right"]))
    mount_rpy_right = os.environ.get("MOUNT_RPY_RIGHT", os.environ.get("MOUNT_RPY", DEFAULT_MOUNT_RPY["right"]))
    mount_yaw_left = os.environ.get("MOUNT_YAW_LEFT", os.environ.get("MOUNT_YAW", DEFAULT_MOUNT_YAW["left"]))
    mount_yaw_right = os.environ.get("MOUNT_YAW_RIGHT", os.environ.get("MOUNT_YAW", DEFAULT_MOUNT_YAW["right"]))
    print(
        "[mount] "
        f"left xyz='{mount_xyz_left}' rpy='{mount_rpy_left}' yaw='{mount_yaw_left}' | "
        f"right xyz='{mount_xyz_right}' rpy='{mount_rpy_right}' yaw='{mount_yaw_right}'"
    )

    # Combine right arm + right hand
    print("Combining right arm + right hand...")
    combined_right = combine_franka_orca(franka, orca_right, "right",
                                          mount_rpy=mount_rpy_right, mount_xyz=mount_xyz_right,
                                          mount_yaw=mount_yaw_right)

    # Combine left arm + left hand
    print("Combining left arm + left hand...")
    combined_left = combine_franka_orca(franka, orca_left, "left",
                                         mount_rpy=mount_rpy_left, mount_xyz=mount_xyz_left,
                                         mount_yaw=mount_yaw_left)

    # Write output URDFs
    right_path = os.path.join(OUTPUT_DIR, "franka_orca_right.urdf")
    left_path = os.path.join(OUTPUT_DIR, "franka_orca_left.urdf")

    xml_header = '<?xml version="1.0" ?>\n'

    with open(right_path, "w") as f:
        f.write(xml_header + prettify_xml(combined_right))
    print(f"Wrote: {right_path}")

    with open(left_path, "w") as f:
        f.write(xml_header + prettify_xml(combined_left))
    print(f"Wrote: {left_path}")

    # Print joint summary
    for name, path in [("Right", right_path), ("Left", left_path)]:
        tree = ET.parse(path)
        root = tree.getroot()
        revolute_joints = [j.get("name") for j in root.iter("joint")
                          if j.get("type") == "revolute"]
        print(f"\n{name} arm+hand ({path}):")
        print(f"  Total revolute joints: {len(revolute_joints)}")
        print(f"  Arm joints (7): {[j for j in revolute_joints if j.startswith('panda_')]}")
        print(f"  Hand joints (17): {[j for j in revolute_joints if not j.startswith('panda_')]}")


if __name__ == "__main__":
    main()
