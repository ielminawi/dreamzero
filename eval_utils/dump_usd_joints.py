"""Dump the physics joint definitions (and joint states) of a USD scene.

Used to compare the colleague's orcav1_franka.usd Franka joint frames against the
standard Panda URDF chain — a constant local-frame rotation about a joint axis in the
USD that is absent in the URDF IS a per-joint zero-offset between the two conventions
(the thing the recorded teleop data is expressed in).

Run inside the Isaac container:
  /opt/IsaacLab/isaaclab.sh -p eval_utils/dump_usd_joints.py --usd /app/orca_v1/orcav1_franka.usd
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str, default="/app/orca_v1/orcav1_franka.usd")
parser.add_argument("--match", type=str, default="", help="substring filter on joint path")
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)   # boots kit so pxr becomes importable
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics, Gf  # noqa: E402

stage = Usd.Stage.Open(args.usd)
print("=" * 100)
print("USD:", args.usd)
print("default prim:", stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None)
print("=" * 100)


def quat_to_axis_angle(q):
    import math
    w = max(-1.0, min(1.0, q.GetReal()))
    im = q.GetImaginary()
    ang = 2 * math.acos(w)
    s = math.sqrt(max(1e-12, 1 - w * w))
    if s < 1e-6:
        return (0, 0, 1), 0.0
    return (im[0] / s, im[1] / s, im[2] / s), ang


n = 0
for prim in stage.Traverse():
    t = prim.GetTypeName()
    if "Joint" not in t:
        continue
    path = str(prim.GetPath())
    if args.match and args.match not in path:
        continue
    n += 1
    j = UsdPhysics.Joint(prim)
    b0 = j.GetBody0Rel().GetTargets()
    b1 = j.GetBody1Rel().GetTargets()
    lp0 = j.GetLocalPos0Attr().Get()
    lr0 = j.GetLocalRot0Attr().Get()
    lp1 = j.GetLocalPos1Attr().Get()
    lr1 = j.GetLocalRot1Attr().Get()
    print(f"\n[{t}] {path}")
    print(f"  body0={[str(x) for x in b0]}  body1={[str(x) for x in b1]}")
    ax0, an0 = quat_to_axis_angle(lr0) if lr0 else ((0, 0, 0), 0)
    ax1, an1 = quat_to_axis_angle(lr1) if lr1 else ((0, 0, 0), 0)
    import math
    print(f"  localPos0={lp0}  localRot0={lr0}  (axis={tuple(round(a,4) for a in ax0)} ang={math.degrees(an0):.2f}deg)")
    print(f"  localPos1={lp1}  localRot1={lr1}  (axis={tuple(round(a,4) for a in ax1)} ang={math.degrees(an1):.2f}deg)")
    if t == "PhysicsRevoluteJoint":
        rj = UsdPhysics.RevoluteJoint(prim)
        print(f"  axis={rj.GetAxisAttr().Get()}  lower={rj.GetLowerLimitAttr().Get()}  upper={rj.GetUpperLimitAttr().Get()}")
    # joint state (rest/initial position) + drive targets if authored
    for attr in prim.GetAttributes():
        nm = attr.GetName()
        if any(k in nm for k in ("state:", "drive:", "physics:jointEnabled")):
            val = attr.Get()
            if val is not None:
                print(f"  {nm} = {val}")
print(f"\ntotal joints: {n}")
