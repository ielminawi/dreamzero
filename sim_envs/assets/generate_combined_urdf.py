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
OUTPUT_DIR = SCRIPT_DIR


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
    ("panda_joint6", "panda_link5", "panda_link6",
     (0, 0, 0), (1.5708, 0, 0), (0, 0, 1), -0.0175, 3.7525),
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


def build_franka_arm_xml() -> ET.Element:
    """Build the Franka Panda arm as an XML element tree (links + joints)."""
    robot = ET.Element("robot")
    robot.set("name", "franka_orca")

    # Material definitions
    for name, rgba in [("white", "1 1 1 1"), ("grey", "0.9 0.9 0.9 1"),
                        ("dark", "0.3 0.3 0.3 1")]:
        mat = ET.SubElement(robot, "material")
        mat.set("name", name)
        color = ET.SubElement(mat, "color")
        color.set("rgba", rgba)

    # Create links with simple cylinder geometry
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

        if radius > 0 and mass > 0:
            # Visual
            visual = ET.SubElement(link, "visual")
            v_origin = ET.SubElement(visual, "origin")
            v_origin.set("xyz", f"0 0 {length / 2 if length > 0 else 0}")
            v_origin.set("rpy", "0 0 0")
            geometry = ET.SubElement(visual, "geometry")
            if length > 0.01:
                cyl = ET.SubElement(geometry, "cylinder")
                cyl.set("radius", str(radius))
                cyl.set("length", str(length))
            else:
                sph = ET.SubElement(geometry, "sphere")
                sph.set("radius", str(radius))
            mat_ref = ET.SubElement(visual, "material")
            mat_ref.set("name", "grey" if "0.9" in color_rgba else "dark")

            # Collision (same as visual)
            collision = ET.SubElement(link, "collision")
            c_origin = ET.SubElement(collision, "origin")
            c_origin.set("xyz", f"0 0 {length / 2 if length > 0 else 0}")
            c_origin.set("rpy", "0 0 0")
            c_geometry = ET.SubElement(collision, "geometry")
            if length > 0.01:
                c_cyl = ET.SubElement(c_geometry, "cylinder")
                c_cyl.set("radius", str(radius))
                c_cyl.set("length", str(length))
            else:
                c_sph = ET.SubElement(c_geometry, "sphere")
                c_sph.set("radius", str(radius))

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

        axis = joint.find("axis")
        if axis is not None:
            xyz = axis.get("xyz", "0 0 0")
            parts = [float(x) for x in xyz.split()]
            if len(parts) == 3:
                parts[0] = -parts[0]  # Negate x-axis component
                axis.set("xyz", f"{parts[0]} {parts[1]} {parts[2]}")

        # Swap joint limits for mirrored joints where axis is negated
        limit = joint.find("limit")
        if limit is not None and axis is not None:
            ax_parts = [float(x) for x in axis.get("xyz", "0 0 0").split()]
            # If the primary axis was negated, swap and negate limits
            orig_axis = joint.find("axis")
            lower = limit.get("lower")
            upper = limit.get("upper")
            if lower and upper:
                l, u = float(lower), float(upper)
                # Only swap if the axis direction was fully reversed
                # (simple heuristic: if original had positive x, now negative)
                # We keep limits as-is since the axis reversal handles direction

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


def combine_franka_orca(franka_robot: ET.Element, orca_robot: ET.Element,
                         side: str, mount_rpy: str = "0 0 0",
                         mount_xyz: str = "0 0 0") -> ET.Element:
    """Combine Franka arm with Orca hand into a single URDF.

    Adds a fixed joint from panda_link8 to the hand's root link.

    Args:
        franka_robot: Franka arm XML tree
        orca_robot: Orca hand XML tree (already has correct prefix)
        side: "left" or "right"
        mount_rpy: RPY rotation of hand relative to Franka flange
        mount_xyz: XYZ offset of hand relative to Franka flange
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

    # Add fixed joint connecting Franka flange to Orca hand root
    mount_joint = ET.SubElement(combined, "joint")
    mount_joint.set("name", f"panda_link8_to_{side}_hand")
    mount_joint.set("type", "fixed")
    ET.SubElement(mount_joint, "parent").set("link", "panda_link8")
    ET.SubElement(mount_joint, "child").set("link", hand_root_link)
    origin = ET.SubElement(mount_joint, "origin")
    origin.set("xyz", mount_xyz)
    origin.set("rpy", mount_rpy)

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

    # Generate left hand by mirroring
    print("Generating left hand by mirroring...")
    orca_left = mirror_orca_hand_to_left(orca_right)
    resolve_mesh_paths(orca_left, orca_description_dir, OUTPUT_DIR)

    # Build Franka arm
    print("Building Franka Panda arm...")
    franka = build_franka_arm_xml()

    # Mounting transform: how the Orca hand attaches to the Franka flange.
    # The Orca hand's root is at its origin; the tower base is offset from root.
    # panda_link8 is the flange face pointing along Z.
    # The hand tower extends upward (Z) from its root.
    # We need to rotate the hand so its Z-axis aligns with the Franka's Z-axis
    # and position it so the tower base sits on the flange.
    #
    # From the Orca URDF: right_world2tower_fixed has xyz="-0.04 0.0 0.04575"
    # meaning the tower base is 4cm back (X) and 4.575cm up (Z) from root.
    # The hand's functional axis points along its Z.
    #
    # Mounting: hand root at the flange origin, no rotation needed (both Z-up).
    # Fine-tuning may be needed after visual inspection in Isaac Sim.
    mount_xyz = "0 0 0"
    mount_rpy = "0 0 0"

    # Combine right arm + right hand
    print("Combining right arm + right hand...")
    combined_right = combine_franka_orca(franka, orca_right, "right",
                                          mount_rpy=mount_rpy, mount_xyz=mount_xyz)

    # Combine left arm + left hand
    print("Combining left arm + left hand...")
    combined_left = combine_franka_orca(franka, orca_left, "left",
                                         mount_rpy=mount_rpy, mount_xyz=mount_xyz)

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
