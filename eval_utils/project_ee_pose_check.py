"""Decisive host-side test: is the recorded EE pose the HAND or the FLANGE?

The 7-dim `observations/qpos_arm_*` rows are [x,y,z, qw,qx,qy,qz] in each arm's
BASE frame (base axes parallel to world, base origin at the configured base pos,
identity base rotation -- the same assumption replay_h5_ik.py and the cfg use).

A recorded POSITION is a single 3D point: whether it denotes the Franka flange
(panda_link8) or the rigidly-mounted Orca HAND, it projects to ONE pixel. So we
can settle the user's hypothesis WITHOUT Isaac / IK / joint angles:

  * project the RAW position p_data into the two calibrated sim cameras,
  * overlay it on the matching REAL frame,
  * score the "darkness" (fraction of near-black pixels = the black Orca hand)
    in a disk around the projected point.

If p_data lands ON the black hand (high darkness, every frame, both cams) the
recorded position IS the hand. If it lands ~8 cm away on the silver wrist/flange
(low darkness) it is the flange. We also draw the recorded ORIENTATION frame
(R/G/B = x/y/z axes of R_data) at that point so the orientation interpretation is
visible in the same picture: if the recorded z-axis points along the visible
finger / palm-normal direction, the orientation is the hand's too.

No Isaac needed:
  ~/.venvs/dz_h5/bin/python eval_utils/project_ee_pose_check.py \
      --frames 0,400,800,1200,1600,2000,2400,2800,3200,3600
"""

import argparse
import os
import re

import cv2
import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(REPO, "sim_envs", "franka_orca_bimanual_cfg.py")

parser = argparse.ArgumentParser()
parser.add_argument("--h5", type=str, default=os.path.join(REPO, "20250826_111157.h5"))
parser.add_argument("--frames", type=str,
                    default="0,400,800,1200,1600,2000,2400,2800,3200,3600")
parser.add_argument("--source", choices=["qpos", "actions"], default="qpos")
parser.add_argument("--out-dir", type=str, default=os.path.join(REPO, "output", "ee_pose_check"))
parser.add_argument("--scan-stride", type=int, default=40,
                    help="aggregate darkness over the whole episode every N frames")
parser.add_argument("--axis-len", type=float, default=0.06, help="orientation axis length (m)")
args = parser.parse_args()


# --------------------------------------------------------------------------- cfg
def parse_cfg(path):
    src = open(path).read()

    def tup(name):
        m = re.search(rf"{name}\s*=\s*\(([^)]*)\)", src)
        return tuple(float(x) for x in m.group(1).split(",") if x.strip())

    sep = float(re.search(r"ARM_SEPARATION_Y\s*=\s*([0-9.]+)", src).group(1))

    def base(name):
        m = re.search(rf"{name}\s*=\s*\(([^)]*)\)", src).group(1)
        sign = -1.0 if "-ARM_SEPARATION_Y" in m else 1.0
        return np.array([0.0, sign * sep / 2, 0.0])

    aria_blk = src[src.index("aria_rgb_cam = CameraCfg"):]
    oakd_blk = src[src.index("oakd_front_view = CameraCfg"):]
    return {
        "aria_eye": tup("_ARIA_EYE"), "aria_rot": tup("_ARIA_ROT"),
        "oakd_eye": tup("_OAKD_EYE"), "oakd_rot": tup("_OAKD_ROT"),
        "left_base": base("_LEFT_BASE_POS"), "right_base": base("_RIGHT_BASE_POS"),
        "aria_focal": float(re.search(r"focal_length\s*=\s*([0-9.]+)", aria_blk).group(1)),
        "oakd_focal": float(re.search(r"focal_length\s*=\s*([0-9.]+)", oakd_blk).group(1)),
        "aperture": 20.955,
    }


C = parse_cfg(CFG)


def quat_to_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


class Cam:
    """Pinhole in the sim OpenGL convention (looks along -Z, +X right, +Y up)."""

    def __init__(self, eye, rot_wxyz, focal, aperture, W, H):
        self.eye = np.asarray(eye, float)
        self.R = quat_to_R(rot_wxyz)
        self.fx = W * focal / aperture
        self.cx, self.cy = W / 2.0, H / 2.0
        self.W, self.H = W, H

    def project(self, pw):
        pc = self.R.T @ (np.asarray(pw, float) - self.eye)
        if pc[2] >= -1e-6:
            return None
        u = self.cx + self.fx * (pc[0] / -pc[2])
        v = self.cy - self.fx * (pc[1] / -pc[2])
        return (float(u), float(v))


ARIA = Cam(C["aria_eye"], C["aria_rot"], C["aria_focal"], C["aperture"], 640, 480)
OAKD = Cam(C["oakd_eye"], C["oakd_rot"], C["oakd_focal"], C["aperture"], 960, 540)


def dark_score(img, uv, r=34):
    """Fraction of near-black pixels (the black Orca hand) in a disk around uv."""
    if uv is None:
        return float("nan")
    H, W = img.shape[:2]
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= u < W and 0 <= v < H):
        return float("nan")
    y0, y1 = max(0, v - r), min(H, v + r)
    x0, x1 = max(0, u - r), min(W, u + r)
    patch = img[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = (yy - v) ** 2 + (xx - u) ** 2 <= r * r
    dark = (patch.mean(-1) < 60) & disk
    return float(dark.sum()) / max(1, disk.sum())


def world_point(p_base, base):
    return base + np.asarray(p_base, float)  # identity base rotation


def draw_pose(img, cam, p_w, R_w, color, tag):
    """Draw the EE point (filled dot) + orientation axes (x=red,y=green,z=blue)."""
    uv = cam.project(p_w)
    if uv is None:
        return None
    u, v = int(round(uv[0])), int(round(uv[1]))
    AX = [((255, 60, 60), R_w[:, 0]), ((60, 255, 60), R_w[:, 1]), ((80, 170, 255), R_w[:, 2])]
    for col, axis in AX:
        tip = cam.project(p_w + args.axis_len * axis)
        if tip is not None:
            cv2.line(img, (u, v), (int(round(tip[0])), int(round(tip[1]))), col, 2, cv2.LINE_AA)
    cv2.circle(img, (u, v), 7, color, -1, cv2.LINE_AA)
    cv2.circle(img, (u, v), 7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, tag, (u + 9, v - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return uv


def lab(im, txt):
    im = np.ascontiguousarray(im)
    cv2.rectangle(im, (0, 0), (im.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(im, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    src = "observations/qpos_arm_{}" if args.source == "qpos" else "actions_arm_{}"
    frames = [int(t) for t in args.frames.split(",")]
    print(f"left base {C['left_base']}  right base {C['right_base']}  source={args.source}")

    GREEN, BLUE = (60, 255, 60), (80, 170, 255)
    rows = []
    with h5py.File(args.h5, "r") as f:
        DL, DR = f[src.format("left")], f[src.format("right")]
        ARIA_DS = f["observations/images/aria_rgb_cam/color"]
        OAKD_DS = f["observations/images/oakd_front_view/color"]
        T = min(len(DL), len(DR), ARIA_DS.shape[0], OAKD_DS.shape[0])

        # ---- aggregate darkness over the whole episode (quantitative verdict) ----
        scan = list(range(0, T, args.scan_stride))
        agg = {"aria_L": [], "aria_R": [], "oakd_L": [], "oakd_R": []}
        for t in scan:
            pl, pr = DL[t], DR[t]
            wL = world_point(pl[0:3], C["left_base"])
            wR = world_point(pr[0:3], C["right_base"])
            for cam, ds, key in [(ARIA, ARIA_DS, "aria"), (OAKD, OAKD_DS, "oakd")]:
                img = ds[t]
                agg[f"{key}_L"].append(dark_score(img, cam.project(wL)))
                agg[f"{key}_R"].append(dark_score(img, cam.project(wR)))
        print(f"\n=== EE-POSITION-on-black-hand darkness over {len(scan)} frames "
              f"(1.0 = fully on the black hand, ~0 = on silver wrist/background) ===")
        for k, v in agg.items():
            v = np.array(v, float); v = v[~np.isnan(v)]
            print(f"  {k:8s}: mean {v.mean():.2f}  median {np.median(v):.2f}  "
                  f">0.5 in {(v > 0.5).mean()*100:4.0f}% of frames  (n={len(v)})")

        # ---- per-frame overlays for visual inspection ----
        print(f"\n=== per-frame overlays -> {args.out_dir} ===")
        for F in frames:
            if F >= T:
                continue
            pl, pr = DL[F], DR[F]
            wL = world_point(pl[0:3], C["left_base"]); RL = quat_to_R(pl[3:7])
            wR = world_point(pr[0:3], C["right_base"]); RR = quat_to_R(pr[3:7])
            aria = np.ascontiguousarray(ARIA_DS[F])
            oakd = np.ascontiguousarray(OAKD_DS[F])

            sc = {}
            for cam, img, nm in [(ARIA, aria, "aria"), (OAKD, oakd, "oakd")]:
                uL = draw_pose(img, cam, wL, RL, GREEN, "L")
                uR = draw_pose(img, cam, wR, RR, BLUE, "R")
                sc[nm] = (dark_score(img, uL), dark_score(img, uR))
            print(f"t={F:4d}  wL={np.array2string(wL,precision=3)} "
                  f"wR={np.array2string(wR,precision=3)}  "
                  f"dark aria L/R={sc['aria'][0]:.2f}/{sc['aria'][1]:.2f}  "
                  f"oakd L/R={sc['oakd'][0]:.2f}/{sc['oakd'][1]:.2f}")

            aria_l = lab(aria, f"REAL aria t={F}  EE-pos dot + RGB=xyz axes (L grn / R blu)")
            oakd_l = lab(cv2.resize(oakd, (640, 360)), f"REAL oakd t={F}  EE-pos")
            pad = np.zeros((480 - 360, 640, 3), np.uint8)
            row = np.concatenate([aria_l, np.concatenate([oakd_l, pad], axis=0)], axis=1)
            cv2.imwrite(os.path.join(args.out_dir, f"ee_pose_t{F}.png"),
                        cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
            rows.append(row)

    if rows:
        sheet = np.concatenate(rows, axis=0)
        cv2.imwrite(os.path.join(args.out_dir, "ee_pose_sheet.png"),
                    cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print("\nwrote", os.path.join(args.out_dir, "ee_pose_sheet.png"))


if __name__ == "__main__":
    main()
