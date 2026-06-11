"""Analytic Franka Panda flange FK + warm-started least-squares IK (shared module).

Convention (verified in docs/EE_POSE_AND_IK_PIPELINE.md): the 7-vector EE pose is
[x, y, z, qx, qy, qz, qw] — flange (panda_link8) position in the arm BASE frame,
quaternion scalar-LAST (XYZW), absolute. The dataset's qpos_arm_*/actions_arm_*
fields use exactly this convention, and the dataset quats live in the qx>0
hemisphere (qw crosses zero, qx in [0.73, 1.0]) — use `canonical_quat_xyzw` so
FK-emitted observations match the training distribution.

IK output joints are directly in the sim URDF convention: LIMITS below match
sim_envs/assets/generate_combined_urdf.py (j4 in [-3.07,-0.07], j6 lower widened
to -1.0). No ARM_SIM_FROM_REAL remap applies to these joints.

Importable both host-side (.venv) and inside the isaac-sim.sif container
(scipy 1.10.1 is bundled).
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

PI = np.pi
PANDA = [{"xyz": [0, 0, 0.333], "rpy": [0, 0, 0]}, {"xyz": [0, 0, 0], "rpy": [-PI / 2, 0, 0]},
         {"xyz": [0, -0.316, 0], "rpy": [PI / 2, 0, 0]}, {"xyz": [0.0825, 0, 0], "rpy": [PI / 2, 0, 0]},
         {"xyz": [-0.0825, 0.384, 0], "rpy": [-PI / 2, 0, 0]}, {"xyz": [0, 0, 0], "rpy": [PI / 2, 0, 0]},
         {"xyz": [0.088, 0, 0], "rpy": [PI / 2, 0, 0]}]
LIMITS = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698],
                   [-2.8973, 2.8973], [-1.0, 3.7525], [-2.8973, 2.8973]])
Q_HOME = np.array([0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741])
A_FIX = [(Rot.from_euler("xyz", j["rpy"]).as_matrix(), np.array(j["xyz"], float)) for j in PANDA]
FLANGE_T = np.array([0.0, 0.0, 0.107])


def fk(q):
    """Flange pose from 7 joint angles -> (R 3x3, p 3) in the arm base frame."""
    R = np.eye(3); p = np.zeros(3)
    for i in range(7):
        Ri, ti = A_FIX[i]; p = p + R @ ti; R = R @ Ri
        c, s = np.cos(q[i]), np.sin(q[i]); R = R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R, p + R @ FLANGE_T


def canonical_quat_xyzw(q4):
    """Resolve the +/-q ambiguity into the dataset's hemisphere (qx > 0)."""
    q4 = np.asarray(q4, dtype=np.float64)
    return -q4 if q4[0] < 0 else q4


def fk_pose7(q):
    """7 joint angles -> [x,y,z,qx,qy,qz,qw] in the dataset convention."""
    R, p = fk(q)
    return np.concatenate([p, canonical_quat_xyzw(Rot.from_matrix(R).as_quat())])


def solve_ik(pose7, q_init, max_nfev=120):
    """EE pose [x,y,z,qx,qy,qz,qw] -> 7 joint angles, warm-started from q_init.

    The quaternion is renormalized here (policy outputs are denormalized per-dim
    and need not be unit norm). Returns (q, pos_err_m, ori_err_rad).
    """
    pose7 = np.asarray(pose7, dtype=np.float64)
    p_t = pose7[0:3]
    quat = pose7[3:7] / max(np.linalg.norm(pose7[3:7]), 1e-8)
    Rt = Rot.from_quat(quat).as_matrix()

    def resid(qq):
        R, p = fk(qq)
        return np.concatenate([p - p_t, Rot.from_matrix(R @ Rt.T).as_rotvec(), 1e-3 * (qq - Q_HOME)])

    q0 = np.clip(np.asarray(q_init, dtype=np.float64), LIMITS[:, 0] + 1e-6, LIMITS[:, 1] - 1e-6)
    sol = least_squares(resid, q0, bounds=(LIMITS[:, 0], LIMITS[:, 1]),
                        xtol=1e-10, ftol=1e-10, max_nfev=max_nfev)
    q = sol.x
    R, p = fk(q)
    pos_err = float(np.linalg.norm(p - p_t))
    ori_err = float(np.linalg.norm(Rot.from_matrix(R @ Rt.T).as_rotvec()))
    return q, pos_err, ori_err


def solve_chunk(poses, q_init, max_nfev=120):
    """IK a whole (T,7) EE-pose chunk, warm-starting each frame from the previous.

    Returns (joints (T,7), pos_err (T,), ori_err (T,)).
    """
    poses = np.asarray(poses, dtype=np.float64)
    T = len(poses)
    out = np.zeros((T, 7)); pe = np.zeros(T); oe = np.zeros(T)
    q = np.asarray(q_init, dtype=np.float64)
    for t in range(T):
        q, pe[t], oe[t] = solve_ik(poses[t], q, max_nfev=max_nfev)
        out[t] = q
    return out, pe, oe
