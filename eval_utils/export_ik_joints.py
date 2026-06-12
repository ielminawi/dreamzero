"""Solve IK OFFLINE for every frame of the episode and export joint trajectories.

Converts the h5 arm EE poses (position = absolute flange position in the arm base
frame; orientation = base-frame rotation DELTA vs the episode-start flange
orientation, see eval_utils/fit_quat_convention.py) into 7-DOF Panda joint
trajectories via warm-started, joint-limit-bounded least-squares IK. The fitted
start orientation A (identical for both arms) makes every frame EXACTLY reachable
(0.0 mm / 0.0 deg residual on validation frames).

Output npz: ik_left/ik_right (T,7) joint trajectories + per-frame residuals.
Play back in Isaac with:  replay_h5_ik.py --joints-npz /app/<out>.npz

Run:  ~/.venvs/dz_h5/bin/python eval_utils/export_ik_joints.py
"""

import argparse
import os
import time

import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PI = np.pi

parser = argparse.ArgumentParser()
parser.add_argument("--h5", type=str, default=os.path.join(REPO, "20250826_111157.h5"))
parser.add_argument("--source", choices=["qpos", "actions"], default="qpos")
parser.add_argument("--out", type=str, default=os.path.join(REPO, "ik_joints.npz"))
# fitted start-orientation A (wxyz), same for both arms (fit_quat_convention.py)
parser.add_argument("--A-left", type=str, default="0.38268 -0.53340 0.53340 0.53340")
parser.add_argument("--A-right", type=str, default="0.38268 -0.53340 0.53340 0.53340")
parser.add_argument("--w-ori", type=float, default=0.3)
# Which body the recorded EE pose is assumed to describe:
#   flange = panda_link8 directly  (pos = p_data, ori = R_data @ A)   [current model]
#   root   = the rigidly-mounted Orca HAND base  (pos = p_data, ori = R_data, RAW)
#            -> mapped to a flange target through the CONSTANT mount transform
#               (panda_link8 -> {side}_gavin_mount[yaw] -> {side}_root[xyz,rpy]),
#               so R_flange = R_data @ R_fr^-1,  p_flange = p_data - R_flange @ t_fr.
# This is the user's "EE pose == hand" hypothesis; A is ignored for root.
parser.add_argument("--target", choices=["flange", "root"], default="flange")
args = parser.parse_args()

# Constant flange->hand-root mount transform per arm, from generate_combined_urdf.py
# (DEFAULT_MOUNT_YAW about flange z, then DEFAULT_MOUNT_XYZ / DEFAULT_MOUNT_RPY).
_MOUNT = {
    "left":  {"yaw": -1.5707963, "rpy": (1.832292, 0.0, 3.141593),
              "xyz": (0.006104277, -0.03057369, 0.08063159)},
    "right": {"yaw": 1.5707963, "rpy": (1.737739, 0.047154, -0.089270),
              "xyz": (0.006220897, 0.03045047, 0.08067889)},
}


def flange_from_root(m):
    """Return (R_fr, t_fr): rotation/translation of the hand root in the flange frame."""
    Rz = Rot.from_euler("z", m["yaw"])
    R_fr = Rz * Rot.from_euler("XYZ", m["rpy"])   # flange -> root rotation
    t_fr = Rz.apply(np.array(m["xyz"]))           # flange -> root translation (flange frame)
    return R_fr.as_matrix(), t_fr

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
    [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.8973, 2.8973],
])
Q_HOME = np.array([0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741])
A_FIX = [(Rot.from_euler("xyz", j["rpy"]).as_matrix(), np.array(j["xyz"], float)) for j in PANDA]
FLANGE_T = np.array([0.0, 0.0, 0.107])


def fk(q):
    R = np.eye(3); p = np.zeros(3)
    for i in range(7):
        Ri, ti = A_FIX[i]
        p = p + R @ ti
        R = R @ Ri
        c, s = np.cos(q[i]), np.sin(q[i])
        R = R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R, p + R @ FLANGE_T


def solve_arm(P, Qd, A_R, name, mount=None):
    """Solve flange IK for each frame. flange model: target = (P, R_data @ A_R).
    root model (mount given): the recorded pose IS the hand root, so map it to a
    flange target through the constant mount transform."""
    T = len(P)
    out = np.zeros((T, 7)); pe = np.zeros(T); oe = np.zeros(T)
    q = Q_HOME.copy()
    R_fr = t_fr = None
    if mount is not None:
        R_fr, t_fr = flange_from_root(mount)
    t0 = time.time()
    for t in range(T):
        Rd = Rot.from_quat(np.r_[Qd[t, 1:4], Qd[t, 0]]).as_matrix()
        if mount is None:
            Rt = Rd @ A_R  # POST composition: R_flange = R_delta @ R_flange(0)
            p_t = P[t]
        else:
            Rt = Rd @ R_fr.T            # R_flange = R_root @ R_fr^-1
            p_t = P[t] - Rt @ t_fr      # p_flange = p_root - R_flange @ t_fr

        def resid(qq):
            R, p = fk(qq)
            return np.concatenate([
                p - p_t,
                args.w_ori * Rot.from_matrix(R @ Rt.T).as_rotvec(),
                1e-3 * (qq - Q_HOME),
            ])

        sol = least_squares(resid, q, bounds=(LIMITS[:, 0], LIMITS[:, 1]),
                            xtol=1e-10, ftol=1e-10, max_nfev=120)
        q = sol.x
        out[t] = q
        R, p = fk(q)
        pe[t] = np.linalg.norm(p - p_t)
        oe[t] = np.linalg.norm(Rot.from_matrix(R @ Rt.T).as_rotvec())
        if t % 800 == 0:
            print(f"  [{name}] {t}/{T}  pos {pe[t]*1000:.2f}mm ori {np.degrees(oe[t]):.2f}deg "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"  [{name}] residuals: pos mean {pe.mean()*1000:.2f}mm max {pe.max()*1000:.2f}mm | "
          f"ori mean {np.degrees(oe.mean()):.2f}deg max {np.degrees(oe.max()):.2f}deg")
    # joint-velocity sanity (50 Hz): real Panda limits ~2.2-2.6 rad/s
    dq = np.abs(np.diff(out, axis=0)) * 50.0
    print(f"  [{name}] max joint speed (rad/s) per joint: {np.round(dq.max(0), 2)}")
    return out, pe, oe


def main():
    with h5py.File(args.h5, "r") as f:
        src = "observations/qpos_arm_{}" if args.source == "qpos" else "actions_arm_{}"
        pl = f[src.format("left")][:]
        pr = f[src.format("right")][:]
    A_L = Rot.from_quat(np.r_[[float(x) for x in args.A_left.split()][1:4],
                              [float(x) for x in args.A_left.split()][0]]).as_matrix()
    A_R = Rot.from_quat(np.r_[[float(x) for x in args.A_right.split()][1:4],
                              [float(x) for x in args.A_right.split()][0]]).as_matrix()
    mount_l = _MOUNT["left"] if args.target == "root" else None
    mount_r = _MOUNT["right"] if args.target == "root" else None
    print(f"solving IK for T={len(pl)} frames, source={args.source}, target={args.target}")
    ql, pel, oel = solve_arm(pl[:, 0:3], pl[:, 3:7], A_L, "left", mount_l)
    qr, per, oer = solve_arm(pr[:, 0:3], pr[:, 3:7], A_R, "right", mount_r)
    np.savez(args.out, ik_left=ql, ik_right=qr,
             pos_err_left=pel, ori_err_left=oel, pos_err_right=per, ori_err_right=oer,
             A_left=args.A_left, A_right=args.A_right, source=args.source)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
