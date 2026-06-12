"""Score POSITION hypotheses by hand-hand separation vs reality.

Hand pose is RIGID to the flange, so palm/fingertip world positions depend ONLY on
the flange pose read from the h5 and the per-arm base placement -- no IK ambiguity.
We therefore build flange world pose = BASE_T @ flange_in_base, then apply the fixed
URDF hand-mount chain (exact transforms from franka_orca_{left,right}.urdf).
"""
import os, numpy as np
import h5py
from scipy.spatial.transform import Rotation as Rot

REPO = "/home/wow/dreamzero"
T_STAR = 1960

def Hm(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; return M
def Hrpy(xyz, rpy):
    return Hm(Rot.from_euler("xyz", rpy).as_matrix(), np.array(xyz, float))

# --- fixed hand-mount chain (exact from branch_collision_analysis.py / URDFs) ---
def hand_chain(side):
    if side == "left":
        g = Hrpy([0, 0, 0], [0, 0, -1.5707963])
        r = Hrpy([0.006220897, -0.03045047, 0.08067889], [1.403854, -0.047154, 3.052323])
        tw = Hrpy([0.04, 0, 0.04575], [0, 0, 0]); w = Hrpy([-0.04063, 0, 0.10487], [0, 0, 0])
        pa = Hrpy([0.002, 0.00144, 0.03872], [0, 0, 0])
    else:
        g = Hrpy([0, 0, 0], [0, 0, 1.5707963])
        r = Hrpy([0.006220897, 0.03045047, 0.08067889], [1.737739, 0.047154, -0.089270])
        tw = Hrpy([-0.04, 0, 0.04575], [0, 0, 0]); w = Hrpy([0.04063, 0, 0.10487], [0, 0, 0])
        pa = Hrpy([0.002, 0.00144, 0.03872], [0, 0, 0])
    P = g @ r @ tw @ w @ pa
    # sample several points across the hand (palm + a fingertip span) for min-dist
    pts = {
        "wrist": g @ r @ tw @ w,
        "palm": P,
        "fingertip": P @ Hm(np.eye(3), np.array([0, 0, 0.09])),
        "finger_side1": P @ Hm(np.eye(3), np.array([0.04, 0, 0.06])),
        "finger_side2": P @ Hm(np.eye(3), np.array([-0.04, 0, 0.06])),
    }
    return pts

HC_L = hand_chain("left"); HC_R = hand_chain("right")

def flange_world_M(pos, quat_xyzw, base_M):
    """base_M @ (flange pose in base)."""
    F = Hm(Rot.from_quat(quat_xyzw).as_matrix(), np.asarray(pos, float))
    return base_M @ F

def hand_pts(pos, quat, base_M, side):
    F = flange_world_M(pos, quat, base_M)
    hc = HC_L if side == "left" else HC_R
    return {k: (F @ M)[:3, 3] for k, M in hc.items()}

def min_hand_dist(ptsL, ptsR):
    return min(np.linalg.norm(a - b) for a in ptsL.values() for b in ptsR.values())

# ---------- load data ----------
with h5py.File(os.path.join(REPO, "20250826_111157.h5"), "r") as f:
    PL = f["observations/qpos_arm_left"][:]   # (T,7) = [x,y,z, qx,qy,qz,qw]
    PR = f["observations/qpos_arm_right"][:]
T = len(PL)
posL, quatL = PL[:, :3], PL[:, 3:]
posR, quatR = PR[:, :3], PR[:, 3:]

calib = None
import json
with open(os.path.join(REPO, "configs/franka_orca_calibration.json")) as fh:
    calib = json.load(fh)
L2R = np.array(calib["left_base_to_right_base"], float)
ARM = 0.6127

def base_pair(name, **kw):
    """Return (leftBaseM, rightBaseM)."""
    sep = kw.get("sep", ARM)
    Lb = Hm(np.eye(3), np.array([0.0, +sep / 2, 0.0]))
    if name == "calib_full":
        # left_base_to_right_base maps a point in right-base frame to left-base frame
        # (T_LR @ p_right = p_left), i.e. it is T^{left}_{right}: right base expressed in
        # left-base coords. cfg says calib Y is flipped vs sim. We mirror the transform
        # about the X-Z plane (Y -> -Y) to convert calib frame -> sim frame, then place
        # right base = left_base_world @ (mirrored L2R).
        S = np.diag([1.0, -1.0, 1.0, 1.0])  # Y-flip
        L2R_sim = S @ L2R @ S
        Rb = Lb @ L2R_sim
        return Lb, Rb
    if name == "calib_noflip":
        Rb = Lb @ L2R
        return Lb, Rb
    if name == "calib_transpose":
        # in case L2R is T^{right}_{left} (left expressed in right coords): invert
        S = np.diag([1.0, -1.0, 1.0, 1.0])
        L2R_sim = S @ np.linalg.inv(L2R) @ S
        return Lb, Lb @ L2R_sim
    # symmetric +-Y, identity rot
    Rb = Hm(np.eye(3), np.array([0.0, -sep / 2, 0.0]))
    return Lb, Rb

def episode_metrics(Lb, Rb, swap_lr=False, neg_y_pos=False):
    pL_, qL_, pR_, qR_ = posL, quatL, posR, quatR
    if swap_lr:
        pL_, qL_, pR_, qR_ = posR, quatR, posL, quatL
    dmins = np.zeros(T)
    cross_R = 0  # right-side hand pts crossing into +Y (left side)
    cross_L = 0  # left-side hand pts crossing into -Y
    for t in range(T):
        plt = pL_[t].copy(); prt = pR_[t].copy()
        if neg_y_pos:
            plt[1] = -plt[1]; prt[1] = -prt[1]
        hL = hand_pts(plt, qL_[t], Lb, "left")
        hR = hand_pts(prt, qR_[t], Rb, "right")
        dmins[t] = min_hand_dist(hL, hR)
        if max(p[1] for p in hR.values()) > 0: cross_R += 1
        if min(p[1] for p in hL.values()) < 0: cross_L += 1
    return dmins, cross_R, cross_L

def frame_detail(Lb, Rb, t, swap_lr=False, neg_y_pos=False):
    pL_, qL_, pR_, qR_ = posL, quatL, posR, quatR
    if swap_lr:
        pL_, qL_, pR_, qR_ = posR, quatR, posL, quatL
    plt = pL_[t].copy(); prt = pR_[t].copy()
    if neg_y_pos:
        plt[1] = -plt[1]; prt[1] = -prt[1]
    hL = hand_pts(plt, qL_[t], Lb, "left")
    hR = hand_pts(prt, qR_[t], Rb, "right")
    return hL, hR, min_hand_dist(hL, hR)

print("=" * 78)
print(f"T={T} frames. Hand pose rigid to flange; hand-hand dist is IK-branch-invariant.")
print(f"h5 flange pos LEFT  t={T_STAR}: {np.round(posL[T_STAR],4)}  quat {np.round(quatL[T_STAR],3)}")
print(f"h5 flange pos RIGHT t={T_STAR}: {np.round(posR[T_STAR],4)}  quat {np.round(quatR[T_STAR],3)}")

# ============ (P1) quick: are L/R positions same-origin-like? ============
print("\n" + "=" * 78)
print("(P1) Are the two arm positions plausibly in a SHARED frame?")
dpos = np.linalg.norm(posL - posR, axis=1)
print(f"  mean |posL-posR| = {dpos.mean():.3f} m  (min {dpos.min():.3f}, max {dpos.max():.3f})")
print(f"  mean posL = {np.round(posL.mean(0),3)}  mean posR = {np.round(posR.mean(0),3)}")
print(f"  posL y range [{posL[:,1].min():.3f},{posL[:,1].max():.3f}]  posR y range [{posR[:,1].min():.3f},{posR[:,1].max():.3f}]")

# ============ CURRENT placement (P0) ============
print("\n" + "=" * 78)
print("CURRENT placement: left +Y, right -Y, identity, sep=0.6127")
Lb0, Rb0 = base_pair("symmetric")
dm0, cR0, cL0 = episode_metrics(Lb0, Rb0)
print(f"  min hand-hand over episode = {dm0.min()*1000:.1f}mm  @ frame {dm0.argmin()}")
print(f"  hand-hand @ t={T_STAR}        = {dm0[T_STAR]*1000:.1f}mm")
print(f"  frames with dmin<0 (interpen) = {(dm0<0).sum()}  ; dmin<30mm = {(dm0<0.03).sum()}  ; <50mm = {(dm0<0.05).sum()}")
print(f"  RIGHT hand crosses midline (y>0) in {cR0}/{T} frames ({100*cR0/T:.1f}%)")
print(f"  LEFT  hand crosses midline (y<0) in {cL0}/{T} frames ({100*cL0/T:.1f}%)")
hL, hR, d = frame_detail(Lb0, Rb0, T_STAR)
print(f"  @t{T_STAR}: L palm {np.round(hL['palm'],3)}  R palm {np.round(hR['palm'],3)}")
print(f"           R max-y {max(p[1] for p in hR.values()):+.3f}  L min-y {min(p[1] for p in hL.values()):+.3f}")

# ============ (P3) calibrated full right-base pose ============
print("\n" + "=" * 78)
print("(P3) FULL calibrated right base (3deg tilt + x/z offset), via Y-flip to sim frame")
for variant in ["calib_full", "calib_noflip", "calib_transpose"]:
    Lb, Rb = base_pair(variant)
    dm, cR, cL = episode_metrics(Lb, Rb)
    print(f"  [{variant:15s}] rightBase pos={np.round(Rb[:3,3],4)}  "
          f"yaw={np.degrees(Rot.from_matrix(Rb[:3,:3]).as_euler('xyz')[2]):+.2f}deg")
    print(f"      min dmin {dm.min()*1000:6.1f}mm  @t{T_STAR} {dm[T_STAR]*1000:6.1f}mm  "
          f"Rcross {100*cR/T:5.1f}%  Lcross {100*cL/T:5.1f}%  interpen(<0) {(dm<0).sum()}")

# ============ (P4) sweep separation ============
print("\n" + "=" * 78)
print("(P4) Sweep ARM_SEPARATION_Y (symmetric, identity rot)")
print(f"  {'sep(m)':>7} {'min_dmin':>9} {'@t1960':>8} {'Rcross%':>8} {'Lcross%':>8} {'interpen':>9}")
for sep in np.arange(0.55, 0.751, 0.025):
    Lb, Rb = base_pair("symmetric", sep=sep)
    dm, cR, cL = episode_metrics(Lb, Rb)
    print(f"  {sep:7.3f} {dm.min()*1000:8.1f}m {dm[T_STAR]*1000:7.1f}m {100*cR/T:7.1f}% {100*cL/T:7.1f}% {(dm<0).sum():9d}")

# ============ (P2) swap L/R and/or negate y ============
print("\n" + "=" * 78)
print("(P2) Swap L/R position data and/or negate in-base EE y")
Lb, Rb = base_pair("symmetric")
for swap in [False, True]:
    for negy in [False, True]:
        dm, cR, cL = episode_metrics(Lb, Rb, swap_lr=swap, neg_y_pos=negy)
        tag = f"swap={swap} negY={negy}"
        print(f"  [{tag:22s}] min {dm.min()*1000:6.1f}mm @t{T_STAR} {dm[T_STAR]*1000:6.1f}mm  "
              f"Rcross {100*cR/T:5.1f}%  Lcross {100*cL/T:5.1f}%  interpen {(dm<0).sum()}")
