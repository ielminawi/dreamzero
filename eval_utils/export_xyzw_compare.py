"""Solve flange IK for several frames under exactly two orientation candidates:
xyzw (scalar-last reading) and its conjugate xyzw_conj. Position taken as-is.

Output npz: frames (F,), ik_left/ik_right (F,2,7) [cand0=xyzw, cand1=xyzw_conj],
residual (F,2,4) [Lpos_mm,Lori_deg,Rpos_mm,Rori_deg], cand_labels.

Run: ~/.venvs/dz_h5/bin/python eval_utils/export_xyzw_compare.py --frames 360,1280,1480,2240,2800,3640
"""
import argparse, os
import h5py, numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PI = np.pi
parser = argparse.ArgumentParser()
parser.add_argument("--h5", type=str, default=os.path.join(REPO, "20250826_111157.h5"))
parser.add_argument("--frames", type=str, default="360,1280,1480,2240,2800,3640")
parser.add_argument("--out", type=str, default=os.path.join(REPO, "xyzw_compare.npz"))
args = parser.parse_args()

PANDA = [{"xyz":[0,0,0.333],"rpy":[0,0,0]},{"xyz":[0,0,0],"rpy":[-PI/2,0,0]},
         {"xyz":[0,-0.316,0],"rpy":[PI/2,0,0]},{"xyz":[0.0825,0,0],"rpy":[PI/2,0,0]},
         {"xyz":[-0.0825,0.384,0],"rpy":[-PI/2,0,0]},{"xyz":[0,0,0],"rpy":[PI/2,0,0]},
         {"xyz":[0.088,0,0],"rpy":[PI/2,0,0]}]
LIMITS = np.array([[-2.8973,2.8973],[-1.7628,1.7628],[-2.8973,2.8973],[-3.0718,-0.0698],
                   [-2.8973,2.8973],[-1.0,3.7525],[-2.8973,2.8973]])
Q_HOME = np.array([0.0,-0.569,0.0,-2.81,0.0,3.037,0.741])
A_FIX = [(Rot.from_euler("xyz",j["rpy"]).as_matrix(), np.array(j["xyz"],float)) for j in PANDA]
FLANGE_T = np.array([0.0,0.0,0.107])

def fk(q):
    R=np.eye(3); p=np.zeros(3)
    for i in range(7):
        Ri,ti=A_FIX[i]; p=p+R@ti; R=R@Ri
        c,s=np.cos(q[i]),np.sin(q[i]); R=R@np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return R, p+R@FLANGE_T

def solve(Rt,pt):
    def resid(qq):
        R,p=fk(qq)
        return np.concatenate([p-pt, Rot.from_matrix(R@Rt.T).as_rotvec(), 1e-3*(qq-Q_HOME)])
    sol=least_squares(resid,Q_HOME.copy(),bounds=(LIMITS[:,0],LIMITS[:,1]),xtol=1e-12,ftol=1e-12,max_nfev=400)
    R,p=fk(sol.x)
    return sol.x, np.linalg.norm(p-pt)*1000, np.degrees(np.linalg.norm(Rot.from_matrix(R@Rt.T).as_rotvec()))

def cands(v):
    a,b,c,d=v[3:7]
    Q=Rot.from_quat([a,b,c,d])      # xyzw
    return [("xyzw",Q),("xyzw_conj",Q.inv())]

def main():
    frames=[int(x) for x in args.frames.split(",")]
    with h5py.File(args.h5,"r") as f:
        L=f["observations/qpos_arm_left"][:].astype(float); R=f["observations/qpos_arm_right"][:].astype(float)
    F=len(frames)
    ikL=np.zeros((F,2,7)); ikR=np.zeros((F,2,7)); resid=np.zeros((F,2,4))
    for fi,Fr in enumerate(frames):
        for ci,((lab,QL),(_,QR)) in enumerate(zip(cands(L[Fr]),cands(R[Fr]))):
            for side,v,Q,col in [("L",L[Fr],QL,0),("R",R[Fr],QR,2)]:
                q,pe,oe=solve(Q.as_matrix(), v[0:3])
                (ikL if side=="L" else ikR)[fi,ci]=q
                resid[fi,ci,col]=pe; resid[fi,ci,col+1]=oe
        print(f"t={Fr:4d}  xyzw: L{resid[fi,0,0]:.0f}mm/{resid[fi,0,1]:.0f}d R{resid[fi,0,2]:.0f}mm/{resid[fi,0,3]:.0f}d | "
              f"conj: L{resid[fi,1,0]:.0f}mm/{resid[fi,1,1]:.0f}d R{resid[fi,1,2]:.0f}mm/{resid[fi,1,3]:.0f}d")
    np.savez(args.out, frames=np.array(frames), ik_left=ikL, ik_right=ikR, residual=resid,
             cand_labels=np.array(["xyzw","xyzw_conj"]))
    print("wrote", args.out)

if __name__=="__main__":
    main()
