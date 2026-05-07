from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.stage import open_stage
import omni.usd
from pxr import Sdf, Usd, UsdGeom, UsdPhysics


SOURCE_USD = "/home/ielminawi/setup/dual_franka_table_manual_rotaed.usda"
OUTPUT_USD = "/home/ielminawi/setup/dual_franka_mounted_visual.usda"

SIDES = {
    "Left": {
        "franka_link8": "/World/Franka_Left/panda_link8",
        "old_gavin": "/World/Gavin_Left",
        "old_orca": "/World/Orca_Left",
    },
    "Right": {
        "franka_link8": "/World/Franka_Right/panda_link8",
        "old_gavin": "/World/Gavin_Right",
        "old_orca": "/World/Orca_Right",
    },
}


def get_stage():
    return omni.usd.get_context().get_stage()


def get_world_matrix(stage, path, cache):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim: {path}")
    return cache.GetLocalToWorldTransform(prim)


def set_local_matrix(stage, path, matrix):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim after reference add: {path}")

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(matrix)


def copy_manual_prim(stage, source_path, target_path):
    root_layer = stage.GetRootLayer()
    source = Sdf.Path(source_path)
    target = Sdf.Path(target_path)

    if stage.GetPrimAtPath(target_path).IsValid():
        stage.RemovePrim(target_path)

    if not Sdf.CopySpec(root_layer, source, root_layer, target):
        raise RuntimeError(f"Could not copy {source_path} to {target_path}")


def remove_physics_under(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return

    to_remove = []
    for prim in Usd.PrimRange(root):
        type_name = prim.GetTypeName()
        if "Joint" in type_name:
            to_remove.append(prim.GetPath())
            continue

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body = UsdPhysics.RigidBodyAPI(prim)
            rigid_body.CreateRigidBodyEnabledAttr(False)
            if hasattr(rigid_body, "CreateKinematicEnabledAttr"):
                rigid_body.CreateKinematicEnabledAttr(False)

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision = UsdPhysics.CollisionAPI(prim)
            collision.CreateCollisionEnabledAttr(False)

    for path in to_remove:
        stage.RemovePrim(path)


def remove_old_manual_parts(stage):
    for side in SIDES.values():
        for path in [side["old_gavin"], side["old_orca"]]:
            if stage.GetPrimAtPath(path).IsValid():
                stage.RemovePrim(path)

    old_bad_joint = "/World/Franka_Right/panda_link7/geometry/ID1683_001frA/RevoluteJoint"
    if stage.GetPrimAtPath(old_bad_joint).IsValid():
        stage.RemovePrim(old_bad_joint)


def mount_visual_parts(stage):
    cache = UsdGeom.XformCache()

    saved_world = {}
    for label, side in SIDES.items():
        saved_world[label] = {
            "link8": get_world_matrix(stage, side["franka_link8"], cache),
            "gavin": get_world_matrix(stage, side["old_gavin"], cache),
            "orca": get_world_matrix(stage, side["old_orca"], cache),
        }

    for label, side in SIDES.items():
        franka_link8 = side["franka_link8"]
        mounted_gavin = f"{franka_link8}/Mounted_Gavin"
        mounted_orca = f"{mounted_gavin}/Mounted_Orca"

        copy_manual_prim(stage, side["old_gavin"], mounted_gavin)
        gavin_local = (
            saved_world[label]["gavin"]
            * saved_world[label]["link8"].GetInverse()
        )
        set_local_matrix(stage, mounted_gavin, gavin_local)

        copy_manual_prim(stage, side["old_orca"], mounted_orca)
        orca_local = (
            saved_world[label]["orca"]
            * saved_world[label]["gavin"].GetInverse()
        )
        set_local_matrix(stage, mounted_orca, orca_local)

        remove_physics_under(stage, mounted_gavin)
        remove_physics_under(stage, mounted_orca)
        print(f"Mounted Gavin and ORCA under {franka_link8}")

    remove_old_manual_parts(stage)


open_stage(SOURCE_USD)
stage = get_stage()

mount_visual_parts(stage)

stage.GetRootLayer().Export(OUTPUT_USD)
print(f"Saved clean mounted visual USD to {OUTPUT_USD}")
print("This scene is visual-only: use the viewport/update loop, not PhysX Play.")

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
