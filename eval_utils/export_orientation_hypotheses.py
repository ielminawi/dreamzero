"""Solve exact IK for ONE frame under many orientation-convention hypotheses.

Position is taken as-is from the recorded EE pose (it already lands correctly).
Only the ORIENTATION definition is varied. For each candidate we build the target
body orientation, map it to a flange target, and solve joint-limited least-squares
IK so the body lands at (recorded position, candidate orientation).

Two attachment models:
  A flange : recorded orientation = panda_link8 orientation       (target body = flange)
  B hand   : recorded orientation = {side}_root (Orca hand) orient (target body = root,
             mapped to a flange target through the constant mount transform)

Output npz: ik_left/ik_right (H,7) joints, labels, residuals, model ('flange'|'root').
Each row h corresponds to labels[h]. Render with render_orientation_hypotheses.py.

Run:  ~/.venvs/dz_h5/bin/python eval_utils/export_orientation_hypotheses.py --frame 1400 --model flange
      ~/.venvs/dz_h5/bin/python eval_utils/export_orientation_hypotheses.py --frame 1400 --model root
"""

import argparse
import os

import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PI = np.pi

parser = argparse.ArgumentParser()
parser.add_argument("--h5", type=str, default=os.path.join(REPO, "20250826_111157.h5"))
parser.add_argument("--frame", type=int, default=1400)
parser.add_argument("--model", choices=["flange", "root"], default="flange")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--A", type=str, default="0.38268 -0.53340 0.53340 0.53340",
                    help="fitted episode-start orientation A0 (wxyz) for the delta models")
parser.add_argument("--w-ori", type=float, default=1.0,
                    help="orientation weight: 1.0 makes orientation the priority (position is "
                         "already known good) so residuals report true reachability of the orient")
args = parser.parse_args()
if args.out is None:
    args.out = os.path.join(REPO, f"ori_hyp_{args.model}_t{args.frame}.npz")

# ---- Panda FK (matches generate_combined_urdf.py / export_ik_joints.py) ----
PANDA = [
    {"xyz": [0, 0, 0.333],       "rpy": [0, 0, 0]},
    {"xyz": [0, 0, 0],           "rpy": [-PI / 2, 0, 0]},
    {"xyz": [0, -0.316, 0],      "rpy": [PI / 2, 0, 0]},
    {"xyz": [0.0825, 0, 0],      "rpy": [PI / 2, 0, 0]},
    {"xyz": [-0.0825, 0.384, 0], "rpy": [-PI / 2, 0, 0]},
    {"xyz": [0, 0, 0],           "rpy": [PI / 2, 0, 0]},
    {"xyz": [0.088, 0, 0],       "rpy": [PI / 2, 0, 0]},
]
LIMITS = np.array([
    [-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
    [-3.0718, -0.0698], [-2.8973, 2.8973], [-1.0, 3.7525], [-2.8973, 2.8973],
])
Q_HOME = np.array([0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741])
A_FIX = [(Rot.from_euler("xyz", j["rpy"]).as_matrix(), np.array(j["xyz"], float)) for j in PANDA]
FLANGE_T = np.array([0.0, 0.0, 0.107])

# constant flange->hand-root mount transform per arm (generate_combined_urdf.py)
_MOUNT = {
    "left":  {"yaw": -1.5707963, "rpy": (1.832292, 0.0, 3.141593),
              "xyz": (0.006104277, -0.03057369, 0.08063159)},
    "right": {"yaw": 1.5707963, "rpy": (1.737739, 0.047154, -0.089270),
              "xyz": (0.006220897, 0.03045047, 0.08067889)},
}


def mount_fr(m):
    Rz = Rot.from_euler("z", m["yaw"])
    R_fr = (Rz * Rot.from_euler("XYZ", m["rpy"])).as_matrix()
    t_fr = Rz.apply(np.array(m["xyz"]))
    return R_fr, t_fr


def fk(q):
    R = np.eye(3); p = np.zeros(3)
    for i in range(7):
        Ri, ti = A_FIX[i]
        p = p + R @ ti
        R = R @ Ri
        c, s = np.cos(q[i]), np.sin(q[i])
        R = R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R, p + R @ FLANGE_T


def solve(R_flange_target, p_flange_target):
    """Joint-limited LS IK for one flange target. Returns (q, pos_mm, ori_deg)."""
    q = Q_HOME.copy()

    def resid(qq):
        R, p = fk(qq)
        return np.concatenate([
            p - p_flange_target,
            args.w_ori * Rot.from_matrix(R @ R_flange_target.T).as_rotvec(),
            1e-3 * (qq - Q_HOME),
        ])

    sol = least_squares(resid, q, bounds=(LIMITS[:, 0], LIMITS[:, 1]),
                        xtol=1e-12, ftol=1e-12, max_nfev=400)
    R, p = fk(sol.x)
    pe = np.linalg.norm(p - p_flange_target) * 1000.0
    oe = np.degrees(np.linalg.norm(Rot.from_matrix(R @ R_flange_target.T).as_rotvec()))
    return sol.x, pe, oe


def candidates(v, A0):
    """v = stored 4 numbers [a,b,c,d]. Return list of (label, R_body 3x3).

    Complete quaternion-interpretation space (mount/attachment held fixed):
    reading order (wxyz scalar-first vs xyzw scalar-last) x conjugation x the
    constant correction A0 (135deg about (-1,1,1)/sqrt3) composed pre/post,
    plus the A0-inverse compositions on the wxyz reading."""
    a, b, c, d = v
    Q_wxyz = Rot.from_quat([b, c, d, a])     # scalar-first reading
    Q_xyzw = Rot.from_quat([a, b, c, d])     # scalar-last reading
    return [
        ("01_xyzw",          Q_xyzw),          # current committed
        ("02_xyzw_conj",     Q_xyzw.inv()),
        ("03_wxyz",          Q_wxyz),
        ("04_wxyz_conj",     Q_wxyz.inv()),
        ("05_A.wxyz_pre",    A0 * Q_wxyz),
        ("06_wxyz.A_post",   Q_wxyz * A0),
        ("07_A.xyzw_pre",    A0 * Q_xyzw),
        ("08_xyzw.A_post",   Q_xyzw * A0),
        ("09_Ainv.wxyz_pre", A0.inv() * Q_wxyz),
        ("10_wxyz.Ainv_post", Q_wxyz * A0.inv()),
    ]


def main():
    A0 = Rot.from_quat(np.r_[[float(x) for x in args.A.split()][1:4],
                             [float(x) for x in args.A.split()][0]])
    with h5py.File(args.h5, "r") as f:
        L = f["observations/qpos_arm_left"][args.frame].astype(float)
        R = f["observations/qpos_arm_right"][args.frame].astype(float)

    labels = [c[0] for c in candidates(L[3:7], A0)]
    ikL = np.zeros((len(labels), 7)); ikR = np.zeros((len(labels), 7))
    resid = np.zeros((len(labels), 4))  # Lpos_mm, Lori_deg, Rpos_mm, Rori_deg
    print(f"frame {args.frame}  model={args.model}  (orientation reachability per candidate)")
    print(f"{'candidate':14s} | {'L pos/ori':>16s} | {'R pos/ori':>16s}")
    for h, ((lab, RbL), (_, RbR)) in enumerate(zip(candidates(L[3:7], A0), candidates(R[3:7], A0))):
        for side, v, Rb, col in [("left", L, RbL, 0), ("right", R, RbR, 2)]:
            R_body = Rb.as_matrix()
            p = v[0:3]
            if args.model == "flange":
                R_flange, p_flange = R_body, p
            else:  # root: map hand-root target -> flange target through the mount
                R_fr, t_fr = mount_fr(_MOUNT[side])
                R_flange = R_body @ R_fr.T
                p_flange = p - R_flange @ t_fr
            q, pe, oe = solve(R_flange, p_flange)
            (ikL if side == "left" else ikR)[h] = q
            resid[h, col] = pe; resid[h, col + 1] = oe
        print(f"{lab:14s} | {resid[h,0]:7.1f}mm {resid[h,1]:5.1f}d | "
              f"{resid[h,2]:7.1f}mm {resid[h,3]:5.1f}d")

    np.savez(args.out, ik_left=ikL, ik_right=ikR, labels=np.array(labels),
             residual=resid, model=args.model, frame=args.frame)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
