"""Fit the ORIENTATION convention of the h5 arm EE poses (host-only, no Isaac).

Established: the 7-dim arm fields are EE poses [x,y,z, qw,qx,qy,qz] with ABSOLUTE
base-frame positions. The QUATERNION is NOT an absolute body orientation though:
both arms read ~identity at frame 0 (qw=0.998/0.993) -- two different physical
wrist poses cannot both be identity, so the quat is a DELTA from the episode-start
(calibration) orientation. The unknown is the constant start orientation A and the
delta composition side:
    PRE :  R_flange(t) = R_A @ R_data(t)      (delta in the base frame ... fixed-axis)
    POST:  R_flange(t) = R_data(t) @ R_A      (delta in the start-pose local frame)
(note: which name maps to which physical meaning depends on convention; both fitted)

Method: for each (arm, model): coarse grid over A in SO(3) -> for each candidate,
track the episode subsample with joint-limit-bounded least-squares IK on the REAL
Panda limits (warm-started frame to frame); score = mean residual. Refine the top
candidates over rotvec(A). Validate the winner on held-out pose-diverse frames.

A near-zero residual for exactly one (model, A) pins the convention; the result is
then verified visually in the Isaac IK replay (eval_utils/replay_h5_ik.py
--quat-pre-*/--quat-post-*).

Run:  ~/.venvs/dz_h5/bin/python eval_utils/fit_quat_convention.py
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
parser.add_argument("--source", choices=["qpos", "actions"], default="qpos")
parser.add_argument("--n-fit", type=int, default=14, help="fit frames (subsampled)")
parser.add_argument("--val-frames", type=str, default="66,228,1362,1550,1809,2313,3264,3417,3605,3826")
parser.add_argument("--w-ori", type=float, default=0.3, help="orientation residual weight (rad->m)")
args = parser.parse_args()

# ---------------- Panda FK (same tables as search_arm_convention.py) ----------------
PANDA = [
    {"xyz": [0, 0, 0.333],       "rpy": [0, 0, 0]},
    {"xyz": [0, 0, 0],           "rpy": [-PI / 2, 0, 0]},
    {"xyz": [0, -0.316, 0],      "rpy": [PI / 2, 0, 0]},
    {"xyz": [0.0825, 0, 0],      "rpy": [PI / 2, 0, 0]},
    {"xyz": [-0.0825, 0.384, 0], "rpy": [-PI / 2, 0, 0]},
    {"xyz": [0, 0, 0],           "rpy": [PI / 2, 0, 0]},
    {"xyz": [0.088, 0, 0],       "rpy": [PI / 2, 0, 0]},
]
# REAL Panda limits (the robot that recorded the data; NOT the widened sim j6)
LIMITS = np.array([
    [-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
    [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.8973, 2.8973],
])
Q_HOME = np.array([0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741])


def _rpy(rpy):
    return Rot.from_euler("xyz", rpy).as_matrix()


A_FIX = [(_rpy(j["rpy"]), np.array(j["xyz"], float)) for j in PANDA]
FLANGE_T = np.array([0.0, 0.0, 0.107])


def fk(q):
    """Flange pose (R, p) in the arm base frame for 7 joint angles."""
    R = np.eye(3); p = np.zeros(3)
    for i in range(7):
        Ri, ti = A_FIX[i]
        p = p + R @ ti
        R = R @ Ri
        c, s = np.cos(q[i]), np.sin(q[i])
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        R = R @ Rz
    p = p + R @ FLANGE_T
    return R, p


def ik(target_R, target_p, q0):
    """Bounded least-squares IK. Returns (q, pos_err_m, ori_err_rad)."""
    def resid(q):
        R, p = fk(q)
        e_p = p - target_p
        e_o = Rot.from_matrix(R @ target_R.T).as_rotvec()
        reg = 1e-3 * (q - Q_HOME)  # fix the redundant DOF, negligible weight
        return np.concatenate([e_p, args.w_ori * e_o, reg])
    sol = least_squares(resid, np.clip(q0, LIMITS[:, 0], LIMITS[:, 1]),
                        bounds=(LIMITS[:, 0], LIMITS[:, 1]), xtol=1e-10, ftol=1e-10, max_nfev=200)
    R, p = fk(sol.x)
    pos_err = float(np.linalg.norm(p - target_p))
    ori_err = float(np.linalg.norm(Rot.from_matrix(R @ target_R.T).as_rotvec()))
    return sol.x, pos_err, ori_err


def track(A_R, model, P, Qd):
    """Track frames with warm-started IK. Returns (mean_pos_err, mean_ori_err, errs)."""
    q = Q_HOME.copy()
    pes, oes = [], []
    for p_t, qd in zip(P, Qd):
        Rd = Rot.from_quat(np.r_[qd[1:4], qd[0]]).as_matrix()  # wxyz -> scipy xyzw
        Rt = A_R @ Rd if model == "PRE" else Rd @ A_R
        q, pe, oe = ik(Rt, p_t, q)
        pes.append(pe); oes.append(oe)
    return float(np.mean(pes)), float(np.mean(oes)), (np.array(pes), np.array(oes))


def grid_rotations():
    """Coarse SO(3) grid: identity + {13 axes} x {45,90,135,180 deg}."""
    axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]
    rots = [Rot.identity()]
    for ax in axes:
        a = np.array(ax, float); a /= np.linalg.norm(a)
        for ang in (PI / 4, PI / 2, 3 * PI / 4, PI, -PI / 4, -PI / 2, -3 * PI / 4):
            rots.append(Rot.from_rotvec(a * ang))
    return rots


def wxyz(R):
    x, y, z, w = R.as_quat()
    s = 1.0 if w >= 0 else -1.0
    return np.array([w, x, y, z]) * s


def main():
    with h5py.File(args.h5, "r") as f:
        src = "observations/qpos_arm_{}" if args.source == "qpos" else "actions_arm_{}"
        data = {s: f[src.format(s)][:] for s in ("left", "right")}
    T = len(data["left"])
    fit_idx = np.unique(np.linspace(0, T - 1, args.n_fit).astype(int))
    val_idx = np.array([int(x) for x in args.val_frames.split(",")])
    print(f"T={T}  fit frames={list(fit_idx)}  val frames={list(val_idx)}")

    for side in ("left", "right"):
        arr = data[side]
        P_fit, Q_fit = arr[fit_idx, 0:3], arr[fit_idx, 3:7]
        P_val, Q_val = arr[val_idx, 0:3], arr[val_idx, 3:7]
        print(f"\n================ {side.upper()} ARM ================")
        # baseline: treat dataset quat as ABSOLUTE flange orientation
        pe, oe, _ = track(np.eye(3), "PRE", P_fit, Q_fit)
        print(f"  baseline ABS (A=I): pos {pe*1000:7.1f} mm   ori {np.degrees(oe):6.1f} deg")

        results = []
        for model in ("PRE", "POST"):
            cands = []
            for Rc in grid_rotations():
                pe, oe, _ = track(Rc.as_matrix(), model, P_fit, Q_fit)
                cands.append((pe + args.w_ori * oe, pe, oe, Rc))
            cands.sort(key=lambda c: c[0])
            # refine top 3 over rotvec(A)
            best = None
            for _, _, _, Rc in cands[:3]:
                def outer(rv):
                    pe, oe, _ = track(Rot.from_rotvec(rv).as_matrix(), model, P_fit, Q_fit)
                    return [pe, args.w_ori * oe]
                sol = least_squares(outer, Rc.as_rotvec(), diff_step=0.02, max_nfev=60)
                Rf = Rot.from_rotvec(sol.x)
                pe, oe, _ = track(Rf.as_matrix(), model, P_fit, Q_fit)
                score = pe + args.w_ori * oe
                if best is None or score < best[0]:
                    best = (score, pe, oe, Rf)
            score, pe, oe, Rf = best
            vpe, voe, (vp, vo) = track(Rf.as_matrix(), model, P_val, Q_val)
            q = wxyz(Rf)
            results.append((score, model, q, pe, oe, vpe, voe, vp, vo))
            print(f"  {model}: fit pos {pe*1000:7.1f} mm  ori {np.degrees(oe):6.1f} deg | "
                  f"VAL pos {vpe*1000:7.1f} mm  ori {np.degrees(voe):6.1f} deg | "
                  f"A(wxyz) = [{q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}]")

        results.sort(key=lambda r: r[0])
        s_, model, q, pe, oe, vpe, voe, vp, vo = results[0]
        flag = "--quat-pre" if model == "PRE" else "--quat-post"
        print(f"  WINNER {model}: val per-frame pos(mm) {np.round(vp*1000,1)}")
        print(f"                  val per-frame ori(deg) {np.round(np.degrees(vo),1)}")
        print(f"  ==> {flag}-{side} \"{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}\"")


if __name__ == "__main__":
    main()
