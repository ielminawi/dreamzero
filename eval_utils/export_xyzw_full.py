"""Full-episode flange IK under the CONFIRMED convention: position as-is, orientation =
quaternion read XYZW (scalar-last), absolute. Warm-started for a smooth, continuous
joint trajectory. Output npz is replay_h5_ik-compatible (ik_left/ik_right + residuals).

Run:  ~/.venvs/dz_h5/bin/python eval_utils/export_xyzw_full.py --out ik_xyzw_full.npz
Then: replay_h5_ik.py --joints-npz /app/ik_xyzw_full.npz
"""
import argparse, os, time
import h5py, numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PI = np.pi
parser = argparse.ArgumentParser()
parser.add_argument("--h5", type=str, default=os.path.join(REPO, "20250826_111157.h5"))
parser.add_argument("--source", choices=["qpos", "actions"], default="qpos")
parser.add_argument("--out", type=str, default=os.path.join(REPO, "ik_xyzw_full.npz"))
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

def solve_arm(P, Q4, name):
    T=len(P); out=np.zeros((T,7)); pe=np.zeros(T); oe=np.zeros(T); q=Q_HOME.copy(); t0=time.time()
    for t in range(T):
        a,b,c,d=Q4[t]; Rt=Rot.from_quat([a,b,c,d]).as_matrix()  # XYZW absolute
        p_t=P[t]
        def resid(qq):
            R,p=fk(qq)
            return np.concatenate([p-p_t, Rot.from_matrix(R@Rt.T).as_rotvec(), 1e-3*(qq-Q_HOME)])
        sol=least_squares(resid,q,bounds=(LIMITS[:,0],LIMITS[:,1]),xtol=1e-10,ftol=1e-10,max_nfev=120)
        q=sol.x; out[t]=q; R,p=fk(q)
        pe[t]=np.linalg.norm(p-p_t); oe[t]=np.linalg.norm(Rot.from_matrix(R@Rt.T).as_rotvec())
        if t%800==0: print(f"  [{name}] {t}/{T} pos {pe[t]*1000:.2f}mm ori {np.degrees(oe[t]):.2f}deg ({time.time()-t0:.0f}s)",flush=True)
    dq=np.abs(np.diff(out,axis=0))*50.0
    print(f"  [{name}] resid pos max {pe.max()*1000:.2f}mm ori max {np.degrees(oe.max()):.2f}deg | max jspd {np.round(dq.max(0),2)}")
    return out, pe, oe

def main():
    with h5py.File(args.h5,"r") as f:
        src="observations/qpos_arm_{}" if args.source=="qpos" else "actions_arm_{}"
        pl=f[src.format("left")][:]; pr=f[src.format("right")][:]
    print(f"solving XYZW flange IK for T={len(pl)} frames")
    ql,pel,oel=solve_arm(pl[:,0:3], pl[:,3:7], "left")
    qr,per,oer=solve_arm(pr[:,0:3], pr[:,3:7], "right")
    np.savez(args.out, ik_left=ql, ik_right=qr, pos_err_left=pel, ori_err_left=oel,
             pos_err_right=per, ori_err_right=oer, convention="xyzw_abs_flange")
    print("wrote", args.out)

if __name__=="__main__":
    main()
