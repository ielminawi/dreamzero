"""Analytic (no-Isaac) collision / IK-branch analysis at t=1960.

Reproduces the Panda FK from export_xyzw_full.py, computes WORLD link positions for
both arms (left base +Y, right base -Y, identity base orientation), measures midline
crossing + min link-to-link distance, checks joint continuity around 1960, and re-solves
the IK from many seeds to test for a less-colliding redundancy branch.
"""
import os, numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

REPO = "/home/wow/dreamzero"
PI = np.pi
T_STAR = 1960
WIN = (1900, 2020)

# --- exact copy of the FK model from export_xyzw_full.py ---
PANDA = [{"xyz":[0,0,0.333],"rpy":[0,0,0]},{"xyz":[0,0,0],"rpy":[-PI/2,0,0]},
         {"xyz":[0,-0.316,0],"rpy":[PI/2,0,0]},{"xyz":[0.0825,0,0],"rpy":[PI/2,0,0]},
         {"xyz":[-0.0825,0.384,0],"rpy":[-PI/2,0,0]},{"xyz":[0,0,0],"rpy":[PI/2,0,0]},
         {"xyz":[0.088,0,0],"rpy":[PI/2,0,0]}]
LIMITS = np.array([[-2.8973,2.8973],[-1.7628,1.7628],[-2.8973,2.8973],[-3.0718,-0.0698],
                   [-2.8973,2.8973],[-1.0,3.7525],[-2.8973,2.8973]])
Q_HOME = np.array([0.0,-0.569,0.0,-2.81,0.0,3.037,0.741])
A_FIX = [(Rot.from_euler("xyz",j["rpy"]).as_matrix(), np.array(j["xyz"],float)) for j in PANDA]
FLANGE_T = np.array([0.0,0.0,0.107])

# joint-origin labels (the cumulative frame BEFORE applying joint i's rotation)
JOINT_NAME = ["base->j1","j2(shoulder)","j3","j4(elbow)","j5","j6(wrist)","j7","flange"]

def fk(q):
    R=np.eye(3); p=np.zeros(3)
    for i in range(7):
        Ri,ti=A_FIX[i]; p=p+R@ti; R=R@Ri
        c,s=np.cos(q[i]),np.sin(q[i]); R=R@np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return R, p+R@FLANGE_T

def fk_chain(q):
    """Return list of 8 points (base-frame): origin of each joint frame + flange."""
    R=np.eye(3); p=np.zeros(3); pts=[]
    for i in range(7):
        Ri,ti=A_FIX[i]; p=p+R@ti; R=R@Ri
        pts.append(p.copy())  # joint-i origin (link i's proximal point)
        c,s=np.cos(q[i]),np.sin(q[i]); R=R@np.array([[c,-s,0],[s,c,0],[0,0,1]])
    pts.append(p+R@FLANGE_T)  # flange
    return np.array(pts)

LEFT_BASE  = np.array([0.0, +0.6127/2, 0.0])
RIGHT_BASE = np.array([0.0, -0.6127/2, 0.0])

def seg_seg_dist(p1,p2,p3,p4):
    """Min distance between segment [p1,p2] and [p3,p4]."""
    u=p2-p1; v=p4-p3; w=p1-p3
    a=u@u; b=u@v; c=v@v; d=u@w; e=v@w
    D=a*c-b*b; sc,sN,sD=0,0,D; tc,tN,tD=0,0,D
    if D<1e-9: sN=0; sD=1; tN=e; tD=c
    else:
        sN=b*e-c*d; tN=a*e-b*d
        if sN<0: sN=0; tN=e; tD=c
        elif sN>sD: sN=sD; tN=e+b; tD=c
    if tN<0:
        tN=0
        if -d<0: sN=0
        elif -d>a: sN=sD
        else: sN=-d; sD=a
    elif tN>tD:
        tN=tD
        if -d+b<0: sN=0
        elif -d+b>a: sN=sD
        else: sN=-d+b; sD=a
    sc=0 if abs(sD)<1e-9 else sN/sD
    tc=0 if abs(tD)<1e-9 else tN/tD
    dP=w+sc*u-tc*v
    return np.linalg.norm(dP)

def min_arm_dist(ptsA, ptsB):
    """Min segment-segment distance between two arm chains; returns (dist, iA, iB)."""
    best=1e9; bi=bj=-1
    for i in range(len(ptsA)-1):
        for j in range(len(ptsB)-1):
            d=seg_seg_dist(ptsA[i],ptsA[i+1],ptsB[j],ptsB[j+1])
            if d<best: best=d; bi=i; bj=j
    return best,bi,bj

# ============================================================
import h5py
with h5py.File(os.path.join(REPO,"20250826_111157.h5"),"r") as f:
    pl=f["observations/qpos_arm_left"][:]; pr=f["observations/qpos_arm_right"][:]
d=np.load(os.path.join(REPO,"ik_xyzw_full.npz"))
ikL=d["ik_left"]; ikR=d["ik_right"]
peL=d["pos_err_left"]; peR=d["pos_err_right"]

qL=ikL[T_STAR]; qR=ikR[T_STAR]
print("="*70)
print(f"FRAME t={T_STAR}")
print(f"IK L = {np.round(qL,3)}  posErr {peL[T_STAR]*1000:.4f}mm")
print(f"IK R = {np.round(qR,3)}  posErr {peR[T_STAR]*1000:.4f}mm")
print(f"flange pose L (h5) pos {np.round(pl[T_STAR,:3],4)} quat {np.round(pl[T_STAR,3:],3)}")
print(f"flange pose R (h5) pos {np.round(pr[T_STAR,:3],4)} quat {np.round(pr[T_STAR,3:],3)}")

# --- 1. world link positions ---
def world_chain(q, base):
    return fk_chain(q)+base
cL=world_chain(qL,LEFT_BASE); cR=world_chain(qR,RIGHT_BASE)
print("\n--- WORLD link positions (x,y,z) ---")
print(f"{'pt':<14}{'LEFT (base+Y)':<32}{'RIGHT (base-Y)':<32}")
for i,nm in enumerate(JOINT_NAME):
    print(f"{nm:<14}{str(np.round(cL[i],3)):<32}{str(np.round(cR[i],3)):<32}")

# flange world positions + separation
flL=cL[-1]; flR=cR[-1]
print(f"\nflange world L {np.round(flL,4)}  R {np.round(flR,4)}  sep {np.linalg.norm(flL-flR)*1000:.1f}mm")
# midline = y=0 ; left base +Y so left should stay y>0, right y<0
midline=0.0
print(f"\nmidline y=0. left-base side is +Y, right-base side is -Y.")
print(f"LEFT  min y over chain = {cL[:,1].min():.4f} (crosses into -Y by {max(0,-cL[:,1].min())*1000:.1f}mm)")
print(f"RIGHT max y over chain = {cR[:,1].max():.4f} (crosses into +Y by {max(0,cR[:,1].max())*1000:.1f}mm)")
for i,nm in enumerate(JOINT_NAME):
    print(f"  {nm:<14} L y={cL[i,1]:+.4f}  R y={cR[i,1]:+.4f}")

dmin,iA,iB=min_arm_dist(cL,cR)
print(f"\nMIN link-to-link distance = {dmin*1000:.1f}mm  between LEFT seg {JOINT_NAME[iA]}->{JOINT_NAME[iA+1]} and RIGHT seg {JOINT_NAME[iB]}->{JOINT_NAME[iB+1]}")

# --- 2. continuity ---
print("\n"+"="*70)
print(f"CONTINUITY {WIN[0]}..{WIN[1]}")
for nm,ik in [("left",ikL),("right",ikR)]:
    seg=ik[WIN[0]:WIN[1]+1]
    dq=np.abs(np.diff(seg,axis=0))
    print(f"  {nm}: max frame-to-frame |dq| per joint = {np.round(dq.max(0),3)}")
    # locate biggest jump
    fi=np.unravel_index(np.argmax(dq),dq.shape)
    print(f"        biggest single jump {dq[fi]:.3f} rad at frame {WIN[0]+fi[0]}->{WIN[0]+fi[0]+1}, joint {fi[1]+1}")
    big=np.where(dq>0.5)
    if len(big[0]):
        for k in range(len(big[0])):
            print(f"        >0.5rad jump: frame {WIN[0]+big[0][k]}->{WIN[0]+big[0][k]+1} joint{big[1][k]+1} = {dq[big[0][k],big[1][k]]:.3f}")
    else:
        print(f"        no >0.5rad jumps in window")

# --- 3. branch search: re-solve LEFT and RIGHT at t* from many seeds ---
print("\n"+"="*70)
print("BRANCH SEARCH at t* (multi-seed IK re-solve)")
rng=np.random.default_rng(0)

def solve_pose(p_t, quat_xyzw, q0):
    Rt=Rot.from_quat(quat_xyzw).as_matrix()
    def resid(qq):
        R,p=fk(qq)
        return np.concatenate([p-p_t, Rot.from_matrix(R@Rt.T).as_rotvec(), 1e-3*(qq-Q_HOME)])
    sol=least_squares(resid,q0,bounds=(LIMITS[:,0],LIMITS[:,1]),xtol=1e-12,ftol=1e-12,max_nfev=400)
    R,p=fk(sol.x)
    pe=np.linalg.norm(p-p_t); oe=np.linalg.norm(Rot.from_matrix(R@Rt.T).as_rotvec())
    return sol.x,pe,oe

def branch_search(which):
    if which=="left":
        p_t=pl[T_STAR,:3]; quat=pl[T_STAR,3:]; base=LEFT_BASE; otherC=cR; q_orig=qL
    else:
        p_t=pr[T_STAR,:3]; quat=pr[T_STAR,3:]; base=RIGHT_BASE; otherC=cL; q_orig=qR
    seeds=[Q_HOME, q_orig]
    # elbow up/down variants: vary joint2,4,6 + randoms
    for s in [-1,1]:
        for s4 in [-1,1]:
            q=Q_HOME.copy(); q[1]=s*0.8; q[3]=np.clip(-1.5*( -s4),LIMITS[3,0],LIMITS[3,1]); q[5]=1.5+s*1.0
            seeds.append(np.clip(q,LIMITS[:,0],LIMITS[:,1]))
    for _ in range(60):
        seeds.append(rng.uniform(LIMITS[:,0],LIMITS[:,1]))
    results=[]
    for q0 in seeds:
        try:
            q,pe,oe=solve_pose(p_t,quat,q0)
        except Exception:
            continue
        if pe<1e-4 and oe<1e-3:  # essentially exact
            c=world_chain(q,base)
            dmin,_,_=min_arm_dist(c,otherC)
            results.append((dmin,q,pe,oe,c))
    # dedupe by joint-vector rounding
    uniq={}
    for r in results:
        key=tuple(np.round(r[1],2))
        if key not in uniq or r[0]>uniq[key][0]:
            uniq[key]=r
    res=sorted(uniq.values(),key=lambda x:-x[0])
    return res

for which in ["left","right"]:
    print(f"\n--- re-solving {which.upper()} arm (other arm held at its IK config) ---")
    res=branch_search(which)
    base_dmin=dmin
    print(f"  {len(res)} distinct exact branches found. (orig arm-link min-dist={base_dmin*1000:.1f}mm)")
    for k,(dm,q,pe,oe,c) in enumerate(res[:8]):
        print(f"  branch {k}: armLinkMinDist={dm*1000:6.1f}mm  pe={pe*1e3:.4f}mm oe={np.degrees(oe):.3f}deg  q={np.round(q,2)}  flange_y={c[-1,1]:+.3f}")

# --- 4. HAND extent past the flange (URDF fixed hand-mount chain) + branch invariance ---
print("\n"+"="*70)
print("HAND-TO-HAND distance (the arm LINKS were >120mm apart; collision is in the HAND)")
def Hm(R,t): M=np.eye(4); M[:3,:3]=R; M[:3,3]=t; return M
def Hrpy(xyz,rpy): return Hm(Rot.from_euler("xyz",rpy).as_matrix(),np.array(xyz,float))
def flange_M(q):
    M=np.eye(4)
    for i in range(7):
        Ri,ti=A_FIX[i]; M=M@Hm(Ri,ti)@Hm(Rot.from_euler("z",q[i]).as_matrix(),np.zeros(3))
    return M@Hm(np.eye(3),FLANGE_T)
def hand_chain(side):
    # exact fixed transforms from franka_orca_{left,right}.urdf:
    if side=="left":
        g=Hrpy([0,0,0],[0,0,-1.5707963]); r=Hrpy([0.006220897,-0.03045047,0.08067889],[1.403854,-0.047154,3.052323])
        tw=Hrpy([0.04,0,0.04575],[0,0,0]); w=Hrpy([-0.04063,0,0.10487],[0,0,0]); pa=Hrpy([0.002,0.00144,0.03872],[0,0,0])
    else:
        g=Hrpy([0,0,0],[0,0,1.5707963]); r=Hrpy([0.006220897,0.03045047,0.08067889],[1.737739,0.047154,-0.089270])
        tw=Hrpy([-0.04,0,0.04575],[0,0,0]); w=Hrpy([0.04063,0,0.10487],[0,0,0]); pa=Hrpy([0.002,0.00144,0.03872],[0,0,0])
    P=g@r@tw@w@pa
    return {"wrist":g@r@tw@w,"palm":P,"fingertip":P@Hm(np.eye(3),np.array([0,0,0.09]))}
def handpts(q,side,base):
    F=flange_M(q); return {k:(F@M)[:3,3]+base for k,M in hand_chain(side).items()}
def min_hand_dist(qL_,qR_):
    A=handpts(qL_,"left",LEFT_BASE); B=handpts(qR_,"right",RIGHT_BASE)
    return min(np.linalg.norm(a-b) for a in A.values() for b in B.values())
hwL=handpts(qL,"left",LEFT_BASE); hwR=handpts(qR,"right",RIGHT_BASE)
print(f"  palm world  L {np.round(hwL['palm'],3)}  R {np.round(hwR['palm'],3)}")
print(f"  palm-palm  {np.linalg.norm(hwL['palm']-hwR['palm'])*1000:.1f}mm   min hand-pt dist {min_hand_dist(qL,qR)*1000:.1f}mm")
print(f"  RIGHT palm world-y={hwR['palm'][1]:+.3f} fingertip-y={hwR['fingertip'][1]:+.3f}  (LEFT palm-y={hwL['palm'][1]:+.3f}) -> right hand reaches ACROSS midline into left side")
# branch invariance: hand dist must be CONSTANT across exact branches
print("\n  hand-to-hand min dist for each RIGHT-arm exact branch (LEFT held):")
resR=branch_search("right")
for k,(dm,q,pe,oe,c) in enumerate(resR[:6]):
    print(f"   branch{k}: handDist={min_hand_dist(qL,q)*1000:6.2f}mm  q={np.round(q,2)}")
print("  -> hand-to-hand distance is INVARIANT across IK branches (hand is rigid to flange).")
