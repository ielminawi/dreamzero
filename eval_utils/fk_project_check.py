"""Host-side arm-convention check: project Panda FK onto the REAL h5 frames.

No Isaac needed. For each requested h5 frame, take the recorded arm qpos, apply the
configured real->sim remap (parsed live from sim_envs/franka_orca_bimanual_cfg.py),
run Franka FK from the configured base positions, and project shoulder/elbow/wrist/
flange into BOTH sim camera models (eye/rot/lens parsed from the cfg — the exact
pinhole approximation the sim renders with). The projected skeleton is drawn ONTO the
matching real camera frame: if the arm convention is right, the projected flange must
land on the black Orca hand in the real image, in every frame, in both cameras.

A wrong per-joint offset produces a systematic, pose-dependent miss; comparing the
miss direction across frames discriminates WHICH joint is off. (The aria camera is a
fisheye approximated by a wide rectilinear pinhole, so expect growing error toward the
aria image edges; the oak-d (~55 deg HFOV) is the quantitative reference.)

Also prints a per-frame "darkness score": the fraction of near-black pixels (the Orca
hand is black) in a disk around each projected flange — a crude auto-match metric.

Usage (host, no Isaac):
  ~/.venvs/dz_h5/bin/python eval_utils/fk_project_check.py \
      --frames 66,228,1362,1550,1809,2313,3264,3417,3605,3826 \
      [--offset "-0.785,0,0,-3.1416,0,0,0"]   # override the remap to test hypotheses

Outputs to --out-dir (default output/fk_overlay):
  fk_overlay_t<F>.png      per frame: [real aria + skeleton | real oakd + skeleton]
  fk_overlay_sheet.png     all frames stacked
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
parser.add_argument("--frames", type=str, default="66,228,1362,1550,1809,2313,3264,3417,3605,3826")
parser.add_argument("--out-dir", type=str, default=os.path.join(REPO, "output", "fk_overlay"))
parser.add_argument("--offset", type=str, default=None,
                    help="comma list of 7 per-joint offsets (sim = real + offset); "
                         "default: ARM_SIM_FROM_REAL parsed from the cfg")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Parse the live cfg (no isaaclab import on the host)
# ---------------------------------------------------------------------------
def parse_cfg(path):
    src = open(path).read()

    def tup(name):
        m = re.search(rf"{name}\s*=\s*\(([^)]*)\)", src)
        return tuple(float(x) for x in m.group(1).split(",") if x.strip())

    out = {
        "aria_eye": tup("_ARIA_EYE"), "aria_rot": tup("_ARIA_ROT"),
        "oakd_eye": tup("_OAKD_EYE"), "oakd_rot": tup("_OAKD_ROT"),
    }
    sep = float(re.search(r"ARM_SEPARATION_Y\s*=\s*([0-9.]+)", src).group(1))
    # _LEFT_BASE_POS is written in terms of ARM_SEPARATION_Y; eval the +/- halves
    def base(name):
        m = re.search(rf"{name}\s*=\s*\(([^)]*)\)", src).group(1)
        sign = -1.0 if "-ARM_SEPARATION_Y" in m else 1.0
        return (0.0, sign * sep / 2, 0.0)
    out["left_base"] = base("_LEFT_BASE_POS")
    out["right_base"] = base("_RIGHT_BASE_POS")
    m = re.search(r"ARM_SIM_FROM_REAL\s*=\s*\[([^\]]*)\]", src).group(1)
    out["remap"] = np.array(eval(f"[{m}]", {"_math": __import__('math')}), float)
    # lens: first focal_length in each camera block
    aria_blk = src[src.index("aria_rgb_cam = CameraCfg"):]
    oakd_blk = src[src.index("oakd_front_view = CameraCfg"):]
    out["aria_focal"] = float(re.search(r"focal_length\s*=\s*([0-9.]+)", aria_blk).group(1))
    out["oakd_focal"] = float(re.search(r"focal_length\s*=\s*([0-9.]+)", oakd_blk).group(1))
    out["aperture"] = 20.955
    return out


C = parse_cfg(CFG)
OFFSET = np.array([float(x) for x in args.offset.split(",")], float) if args.offset else C["remap"]


# ---------------------------------------------------------------------------
# FK + projection
# ---------------------------------------------------------------------------
PANDA = [  # standard Franka Panda URDF joint frames (matches generate_combined_urdf.py)
    {"xyz": [0, 0, 0.333],       "rpy": [0, 0, 0]},
    {"xyz": [0, 0, 0],           "rpy": [-np.pi / 2, 0, 0]},
    {"xyz": [0, -0.316, 0],      "rpy": [np.pi / 2, 0, 0]},
    {"xyz": [0.0825, 0, 0],      "rpy": [np.pi / 2, 0, 0]},
    {"xyz": [-0.0825, 0.384, 0], "rpy": [-np.pi / 2, 0, 0]},
    {"xyz": [0, 0, 0],           "rpy": [np.pi / 2, 0, 0]},
    {"xyz": [0.088, 0, 0],       "rpy": [np.pi / 2, 0, 0]},
]


def _rpy(rpy):
    rx, ry, rz = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # URDF convention


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def hom(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def franka_fk(q, base):
    """Returns the 9 chain points: base, o1..o7 (joint origins), flange (link8)."""
    T = hom(np.eye(3), np.asarray(base, float))
    pts = [T[:3, 3].copy()]
    for jdef, qi in zip(PANDA, q):
        T = T @ hom(_rpy(jdef["rpy"]), np.array(jdef["xyz"], float)) @ hom(rot_z(qi), np.zeros(3))
        pts.append(T[:3, 3].copy())
    T = T @ hom(np.eye(3), np.array([0, 0, 0.107]))
    pts.append(T[:3, 3].copy())
    return np.array(pts)  # (9,3)


def quat_to_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


class Cam:
    """Pinhole camera in the sim's OpenGL convention (looks along -Z, +X right, +Y up)."""

    def __init__(self, eye, rot_wxyz, focal, aperture, W, H):
        self.eye = np.asarray(eye, float)
        self.R = quat_to_R(rot_wxyz)          # camera -> world
        self.fx = W * focal / aperture        # square pixels (vert aperture scales H/W)
        self.cx, self.cy = W / 2.0, H / 2.0
        self.W, self.H = W, H

    def project(self, pw):
        pc = self.R.T @ (np.asarray(pw, float) - self.eye)
        if pc[2] >= -1e-6:                    # behind the camera
            return None
        u = self.cx + self.fx * (pc[0] / -pc[2])
        v = self.cy - self.fx * (pc[1] / -pc[2])
        return (float(u), float(v))


ARIA = Cam(C["aria_eye"], C["aria_rot"], C["aria_focal"], C["aperture"], 640, 480)
OAKD = Cam(C["oakd_eye"], C["oakd_rot"], C["oakd_focal"], C["aperture"], 960, 540)

# chain indices to draw: base(0) shoulder(2) elbow(4) wrist(6) flange(8)
SKEL = [0, 2, 4, 6, 8]


def draw_skel(img, cam, pts, color):
    uv = [cam.project(p) for p in pts]
    for a, b in zip(SKEL[:-1], SKEL[1:]):
        if uv[a] and uv[b]:
            cv2.line(img, (int(uv[a][0]), int(uv[a][1])), (int(uv[b][0]), int(uv[b][1])),
                     color, 2, cv2.LINE_AA)
    for i in SKEL[1:-1]:
        if uv[i]:
            cv2.circle(img, (int(uv[i][0]), int(uv[i][1])), 5, color, 2, cv2.LINE_AA)
    if uv[8]:  # flange: cross
        u, v = int(uv[8][0]), int(uv[8][1])
        cv2.drawMarker(img, (u, v), color, cv2.MARKER_TILTED_CROSS, 18, 3)
    return uv[8]


def dark_score(img, uv, r=40):
    """Fraction of near-black pixels (the Orca hand) in a disk around uv."""
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


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    frames = [int(t) for t in args.frames.split(",")]
    print(f"remap offset (sim = real + offset): {np.array2string(OFFSET, precision=4)}")
    print(f"left base {C['left_base']}  right base {C['right_base']}")

    GREEN, BLUE = (60, 255, 60), (80, 170, 255)  # left, right (RGB)
    rows = []
    with h5py.File(args.h5, "r") as f:
        for F in frames:
            ql = f["observations/qpos_arm_left"][F] + OFFSET
            qr = f["observations/qpos_arm_right"][F] + OFFSET
            aria = np.ascontiguousarray(f["observations/images/aria_rgb_cam/color"][F])
            oakd = np.ascontiguousarray(f["observations/images/oakd_front_view/color"][F])

            ptsL = franka_fk(ql, C["left_base"])
            ptsR = franka_fk(qr, C["right_base"])

            scores = {}
            for cam, img, nm in [(ARIA, aria, "aria"), (OAKD, oakd, "oakd")]:
                fL = draw_skel(img, cam, ptsL, GREEN)
                fR = draw_skel(img, cam, ptsR, BLUE)
                scores[nm] = (dark_score(img, fL), dark_score(img, fR))

            print(f"t={F:4d}  flangeL_world={np.array2string(ptsL[8], precision=3)} "
                  f"flangeR_world={np.array2string(ptsR[8], precision=3)}  "
                  f"dark@flange aria L/R={scores['aria'][0]:.2f}/{scores['aria'][1]:.2f}  "
                  f"oakd L/R={scores['oakd'][0]:.2f}/{scores['oakd'][1]:.2f}")

            def lab(im, txt):
                im = np.ascontiguousarray(im)
                cv2.rectangle(im, (0, 0), (im.shape[1], 26), (0, 0, 0), -1)
                cv2.putText(im, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                return im
            aria_l = lab(aria, f"REAL aria t={F} + FK skeleton (L=green R=blue)")
            oakd_l = lab(cv2.resize(oakd, (640, 360)), f"REAL oakd t={F} + FK")
            pad = np.zeros((480 - 360, 640, 3), np.uint8)
            row = np.concatenate([aria_l, np.concatenate([oakd_l, pad], axis=0)], axis=1)
            cv2.imwrite(os.path.join(args.out_dir, f"fk_overlay_t{F}.png"),
                        cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
            rows.append(row)

    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(os.path.join(args.out_dir, "fk_overlay_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("wrote", os.path.join(args.out_dir, "fk_overlay_sheet.png"))


if __name__ == "__main__":
    main()
