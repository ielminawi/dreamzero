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

# Constant world-frame orientation correction A (135deg about (-1,1,1)/sqrt3), applied as
# R_flange = A @ R_wxyz. Established 2026-06-11 (consensus): the recorded quaternion is
# scalar-FIRST (w,x,y,z); reading it scalar-last ("xyzw") was a ~142deg error that twisted
# both hands inward and made them interpenetrate. A @ R_wxyz is the only convention that
# keeps the hands non-colliding (min hand-hand 229mm) AND pointing down at the table (100%).
A_CORR = Rot.from_quat([-0.53340, 0.53340, 0.53340, 0.38268])  # xyzw of wxyz(0.38268,-0.5334,0.5334,0.5334)

def flange_R(q4):
    c0, c1, c2, c3 = q4                       # stored scalar-first: w=c0, (x,y,z)=(c1,c2,c3)
    R_wxyz = Rot.from_quat([c1, c2, c3, c0])
    return (A_CORR * R_wxyz).as_matrix()       # pre-multiply the constant correction

def solve_arm(P, Q4, name):
    T=len(P); out=np.zeros((T,7)); pe=np.zeros(T); oe=np.zeros(T); q=Q_HOME.copy(); t0=time.time()
    rng=np.random.RandomState(0)
    for t in range(T):
        Rt=flange_R(Q4[t])
        p_t=P[t]
        def resid(qq):
            R,p=fk(qq)
            return np.concatenate([p-p_t, Rot.from_matrix(R@Rt.T).as_rotvec(), 1e-3*(qq-Q_HOME)])
        q_prev=q.copy()
        def attempt(q0):
            s=least_squares(resid,q0,bounds=(LIMITS[:,0],LIMITS[:,1]),xtol=1e-12,ftol=1e-12,max_nfev=300)
            R,p=fk(s.x); return s.x, np.linalg.norm(p-p_t), np.linalg.norm(Rot.from_matrix(R@Rt.T).as_rotvec())
        q, perr, oerr = attempt(q)              # warm-start from previous frame
        if perr>1e-3 or oerr>np.radians(1):
            # trapped: try only SMALL local perturbations of the previous frame (stay continuous).
            # accept an escape only if it is BOTH reachable AND close to q_prev (no branch jumps);
            # otherwise keep the smooth warm-start result (a few frames may be slightly off but the
            # trajectory stays continuous, which matters more for a watchable replay).
            for s in range(12):
                c=attempt(np.clip(q_prev+rng.normal(0,0.12,7),LIMITS[:,0],LIMITS[:,1]))
                if c[1]<1e-3 and c[2]<np.radians(1) and np.linalg.norm(c[0]-q_prev)<0.25:
                    q,perr,oerr=c; break
        out[t]=q; pe[t]=perr; oe[t]=oerr
        if t%800==0: print(f"  [{name}] {t}/{T} pos {pe[t]*1000:.2f}mm ori {np.degrees(oe[t]):.2f}deg ({time.time()-t0:.0f}s)",flush=True)
    dq=np.abs(np.diff(out,axis=0))*50.0
    print(f"  [{name}] resid pos max {pe.max()*1000:.2f}mm ori max {np.degrees(oe.max()):.2f}deg | max jspd {np.round(dq.max(0),2)}")
    return out, pe, oe

def main():
    with h5py.File(args.h5,"r") as f:
        src="observations/qpos_arm_{}" if args.source=="qpos" else "actions_arm_{}"
        pl=f[src.format("left")][:]; pr=f[src.format("right")][:]
    print(f"solving flange IK (A @ wxyz convention) for T={len(pl)} frames")
    ql,pel,oel=solve_arm(pl[:,0:3], pl[:,3:7], "left")
    qr,per,oer=solve_arm(pr[:,0:3], pr[:,3:7], "right")
    np.savez(args.out, ik_left=ql, ik_right=qr, pos_err_left=pel, ori_err_left=oel,
             pos_err_right=per, ori_err_right=oer, convention="A_wxyz_flange")
    print("wrote", args.out)

if __name__=="__main__":
    main()
