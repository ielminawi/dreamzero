"""Quantify flange-vs-hand reinterpretation of the recorded EE pose w.r.t. the
sim hand collision at t=1960. Analytic Panda FK + Gavin-mount compose. No Isaac."""
import os, numpy as np
from scipy.spatial.transform import Rotation as Rot

REPO = "/home/wow/dreamzero"
PI = np.pi

# ---- Panda FK (copied from export_xyzw_full.py) ----
PANDA = [{"xyz":[0,0,0.333],"rpy":[0,0,0]},{"xyz":[0,0,0],"rpy":[-PI/2,0,0]},
         {"xyz":[0,-0.316,0],"rpy":[PI/2,0,0]},{"xyz":[0.0825,0,0],"rpy":[PI/2,0,0]},
         {"xyz":[-0.0825,0.384,0],"rpy":[-PI/2,0,0]},{"xyz":[0,0,0],"rpy":[PI/2,0,0]},
         {"xyz":[0.088,0,0],"rpy":[PI/2,0,0]}]
A_FIX = [(Rot.from_euler("xyz",j["rpy"]).as_matrix(), np.array(j["xyz"],float)) for j in PANDA]
FLANGE_T = np.array([0.0,0.0,0.107])

def fk_flange(q):
    """Return (R_flange, p_flange) in the ARM-BASE frame."""
    R=np.eye(3); p=np.zeros(3)
    for i in range(7):
        Ri,ti=A_FIX[i]; p=p+R@ti; R=R@Ri
        c,s=np.cos(q[i]),np.sin(q[i]); R=R@np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return R, p+R@FLANGE_T

# ---- Mount chain (from generate_combined_urdf.py) ----
def homog(R,t):
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=t; return T

def rpy_mat(rpy):
    return Rot.from_euler("xyz", rpy).as_matrix()

MOUNT_YAW = {"left": -1.5707963, "right": 1.5707963}
MOUNT_XYZ = {"left": [0.006220897,-0.03045047,0.08067889],
             "right":[0.006220897, 0.03045047,0.08067889]}
MOUNT_RPY = {"left": [1.403854,-0.047154,3.052323],
             "right":[1.737739, 0.047154,-0.089270]}

def mount_T(side):
    """flange(panda_link8) -> {side}_root, as a 4x4."""
    # panda_link8 -> gavin_mount : rpy = (0,0,mount_yaw), xyz 0
    T1 = homog(rpy_mat([0,0,MOUNT_YAW[side]]), np.zeros(3))
    # gavin_mount -> root : xyz=mount_xyz, rpy=mount_rpy
    T2 = homog(rpy_mat(MOUNT_RPY[side]), np.array(MOUNT_XYZ[side]))
    return T1 @ T2

# arm base positions in WORLD
BASE = {"left": np.array([0.0, 0.30635, 0.0]),
        "right":np.array([0.0,-0.30635, 0.0])}

def main():
    d = np.load(os.path.join(REPO,"ik_xyzw_full.npz"))
    qL = d["ik_left"]; qR = d["ik_right"]
    T = len(qL)
    print(f"T={T} frames")

    # recorded EE position in base frame, reconstructed via FK flange (the IK target).
    # We FK the solved joints to get flange world pose, then mount->root.
    ML = mount_T("left"); MR = mount_T("right")

    flangeL = np.zeros((T,3)); flangeR = np.zeros((T,3))
    rootL   = np.zeros((T,3)); rootR   = np.zeros((T,3))
    for t in range(T):
        RLb,pLb = fk_flange(qL[t]); RRb,pRb = fk_flange(qR[t])
        # flange world
        fL = BASE["left"]  + pLb;  fR = BASE["right"] + pRb
        flangeL[t]=fL; flangeR[t]=fR
        # flange world homog (base R = identity world orientation; base only translates)
        TfL = homog(RLb, fL); TfR = homog(RRb, fR)
        TrL = TfL @ ML; TrR = TfR @ MR
        rootL[t]=TrL[:3,3]; rootR[t]=TrR[:3,3]

    # ---- CURRENT (flange) interpretation: root-root distance ----
    dRoot = np.linalg.norm(rootL - rootR, axis=1)
    dFlange = np.linalg.norm(flangeL - flangeR, axis=1)

    # ---- HAND interpretation: root == recorded pos == flange position of data ----
    # (because under the hand-interp we'd place the root at base+pos, i.e. exactly
    #  where we currently place the flange). So hand-interp root-root == dFlange.
    dRoot_handinterp = dFlange

    def report(name, arr):
        i = int(np.argmin(arr))
        print(f"  {name}: @1960={arr[1960]*1000:7.1f}mm  min={arr.min()*1000:7.1f}mm @t={i}  @3638={arr[3638]*1000:7.1f}mm")

    print("\n== FLANGE interpretation (CURRENT sim) ==")
    report("flange-flange dist", dFlange)
    report("ROOT-ROOT dist    ", dRoot)
    print("\n== HAND interpretation (root = recorded pos) ==")
    report("ROOT-ROOT dist    ", dRoot_handinterp)

    # frames where flange-interp root-root is tiny / overlapping
    thr = 0.05
    bad = np.where(dRoot < thr)[0]
    print(f"\nFLANGE-interp root-root < {thr*1000:.0f}mm at {len(bad)} frames"
          + (f"; e.g. {bad[:10]}" if len(bad) else ""))

    # how far inward each hand shifts going hand-interp -> flange-interp.
    # = lateral (Y, toward-other-arm) component of (root - flange) world.
    shiftL_y = (rootL[:,1] - flangeL[:,1])   # left arm at +Y; inward = -Y
    shiftR_y = (rootR[:,1] - flangeR[:,1])   # right arm at -Y; inward = +Y
    print("\n== mount lateral (world Y) shift root-vs-flange @1960 ==")
    print(f"  left  root.y - flange.y = {shiftL_y[1960]*1000:+.1f}mm (inward if -)")
    print(f"  right root.y - flange.y = {shiftR_y[1960]*1000:+.1f}mm (inward if +)")
    print(f"  net Y closure from mount @1960 = {(shiftR_y[1960]-shiftL_y[1960])*1000:+.1f}mm")
    # full 3D mount lateral magnitude
    mlL = np.linalg.norm(rootL[1960]-flangeL[1960]); mlR=np.linalg.norm(rootR[1960]-flangeR[1960])
    print(f"  |root-flange| @1960  left={mlL*1000:.1f}mm  right={mlR*1000:.1f}mm")

    print("\n== positions @1960 (world) ==")
    for nm,a in [("flangeL",flangeL),("flangeR",flangeR),("rootL",rootL),("rootR",rootR)]:
        print(f"  {nm} = {np.round(a[1960],4)}")

if __name__=="__main__":
    main()
