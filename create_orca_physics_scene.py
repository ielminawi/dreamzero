from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.stage import open_stage
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


SOURCE_USD = "/home/ielminawi/setup/dual_franka_table_manual_rotaed.usda"
OUTPUT_USD = "/home/ielminawi/setup/dual_franka_orca_physics.usda"

SIDES = {
    "Left": {
        "franka_base": "/World/Franka_Left/panda_link0",
        "franka_wrist": "/World/Franka_Left/panda_link8",
        "gavin_body": "/World/Gavin_Left/node_/mesh_",
        "orca_base": "/World/Orca_Left/worldBody/worldBody",
        "orca_root_joint": "/World/Orca_Left/joints/rootJoint_worldBody",
    },
    "Right": {
        "franka_base": "/World/Franka_Right/panda_link0",
        "franka_wrist": "/World/Franka_Right/panda_link8",
        "gavin_body": "/World/Gavin_Right/node_/mesh_",
        "orca_base": "/World/Orca_Right/worldBody/worldBody",
        "orca_root_joint": "/World/Orca_Right/joints/rootJoint_worldBody",
    },
}

BAD_MANUAL_JOINTS = [
    "/World/Gavin_Left/RevoluteJoint",
    "/World/Gavin_Left/FixedJoint",
    "/World/Gavin_Right/RevoluteJoint",
    "/World/Gavin_Right/FixedJoint",
    "/World/Gavin_Right/PrismaticJoint",
    "/World/Franka_Right/panda_link7/geometry/ID1683_001frA/RevoluteJoint",
]


def get_stage():
    return omni.usd.get_context().get_stage()


def delete_if_exists(stage, path):
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)
        print(f"Removed {path}")


def deactivate_if_exists(stage, path):
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        prim.SetActive(False)
        print(f"Deactivated {path}")


def ensure_rigid_body(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing rigid body target: {path}")

    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    return prim


def enable_existing_rigid_bodies_under(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"Missing rigid body root: {root_path}")

    enabled = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        rigid_body = UsdPhysics.RigidBodyAPI(prim)
        rigid_body.CreateRigidBodyEnabledAttr(True)
        if hasattr(rigid_body, "CreateKinematicEnabledAttr"):
            rigid_body.CreateKinematicEnabledAttr(False)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        enabled += 1

    print(f"Enabled {enabled} rigid bodies under {root_path}")


def add_world_fixed_joint(stage, joint_path, body_path):
    delete_if_exists(stage, joint_path)

    body = stage.GetPrimAtPath(body_path)
    if not body.IsValid():
        raise RuntimeError(f"Missing world-fixed body: {body_path}")

    world_xform = UsdGeom.XformCache().GetLocalToWorldTransform(body)
    world_pos = world_xform.ExtractTranslation()

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
    joint.CreateBreakForceAttr(float("inf"))
    joint.CreateBreakTorqueAttr(float("inf"))
    joint.CreateLocalPos0Attr(Gf.Vec3f(world_pos))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def add_current_pose_fixed_joint(stage, joint_path, body0_path, body1_path):
    delete_if_exists(stage, joint_path)

    body0 = stage.GetPrimAtPath(body0_path)
    body1 = stage.GetPrimAtPath(body1_path)
    if not body0.IsValid():
        raise RuntimeError(f"Missing joint body0: {body0_path}")
    if not body1.IsValid():
        raise RuntimeError(f"Missing joint body1: {body1_path}")

    cache = UsdGeom.XformCache()
    body0_to_world = cache.GetLocalToWorldTransform(body0)
    body1_to_world = cache.GetLocalToWorldTransform(body1)

    joint_world_pos = body1_to_world.ExtractTranslation()
    local_pos0 = body0_to_world.GetInverse().Transform(joint_world_pos)
    local_pos1 = Gf.Vec3d(0.0, 0.0, 0.0)

    rot0 = body0_to_world.ExtractRotation()
    rot1 = body1_to_world.ExtractRotation()
    local_rot0 = rot0.GetInverse() * rot1
    local_rot0_quat = local_rot0.GetQuat()
    local_rot0_imag = local_rot0_quat.GetImaginary()

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateBreakForceAttr(float("inf"))
    joint.CreateBreakTorqueAttr(float("inf"))
    joint.CreateLocalPos0Attr(Gf.Vec3f(local_pos0))
    joint.CreateLocalRot0Attr(
        Gf.Quatf(
            float(local_rot0_quat.GetReal()),
            Gf.Vec3f(
                float(local_rot0_imag[0]),
                float(local_rot0_imag[1]),
                float(local_rot0_imag[2]),
            ),
        )
    )
    joint.CreateLocalPos1Attr(Gf.Vec3f(local_pos1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def add_orca_demo_drives(stage, root_path, target_degrees=12.0):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return

    for prim in Usd.PrimRange(root):
        if "Joint" not in prim.GetTypeName():
            continue
        if prim.GetName() == "rootJoint_worldBody":
            continue

        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTargetPositionAttr(float(target_degrees))
        drive.CreateStiffnessAttr(250.0)
        drive.CreateDampingAttr(25.0)
        drive.CreateMaxForceAttr(5.0)


open_stage(SOURCE_USD)
stage = get_stage()

for path in BAD_MANUAL_JOINTS:
    delete_if_exists(stage, path)

for label, side in SIDES.items():
    # The ORCA file was exported with a fixed root joint. Remove only that
    # root anchor so the hand base can be fixed to Gavin instead of to world.
    deactivate_if_exists(stage, side["orca_root_joint"])

    ensure_rigid_body(stage, side["gavin_body"])
    enable_existing_rigid_bodies_under(stage, f"/World/Orca_{label}")
    ensure_rigid_body(stage, side["orca_base"])

    add_world_fixed_joint(
        stage,
        f"/World/PhysicsJoints/Fix_Franka_{label}_Base_To_World",
        side["franka_base"],
    )
    add_current_pose_fixed_joint(
        stage,
        f"/World/PhysicsJoints/Fix_Gavin_{label}_To_Franka_Wrist",
        side["franka_wrist"],
        side["gavin_body"],
    )
    add_current_pose_fixed_joint(
        stage,
        f"/World/PhysicsJoints/Fix_Orca_{label}_To_Gavin",
        side["gavin_body"],
        side["orca_base"],
    )

    add_orca_demo_drives(stage, f"/World/Orca_{label}", target_degrees=10.0)
    print(f"Prepared physics-mounted ORCA {label}")

stage.GetRootLayer().Export(OUTPUT_USD)
print(f"Saved ORCA physics scene to {OUTPUT_USD}")
print("The stable visual scene was not touched.")
print("If this scene jitters, lower ORCA drive stiffness or test one hand at a time.")

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
