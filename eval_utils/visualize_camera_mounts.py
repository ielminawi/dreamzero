#!/usr/bin/env python3
"""Visualise where the sim cameras are mounted relative to the two Franka arms.

Unlike debug_camera_transforms.py (which compared 3 placement *hypotheses*), this
reads the *live* camera constants straight out of sim_envs/franka_orca_bimanual_cfg.py
(_ARIA_EYE/_ARIA_ROT/_OAKD_EYE/_OAKD_ROT, ARM_SEPARATION_Y, the home pose, the lens
focal lengths) and renders the ACTUAL current configuration:

  * both arm FK chains at the home pose, the two arm bases, the table
  * each camera as a position marker + axis triad + view frustum (true HFOV/VFOV)
  * the ray from each camera's optical centre to where it pierces the table top

It also re-derives the expected camera pose from configs/franka_orca_calibration.json
(pos = arm_base + [tx, -ty, tz]; forward = calib +Z) and prints the residual between
the cfg and calibration, so "are the cameras still shifted relative to the arms?" gets
a numeric answer, not just a picture.

Pure host script (numpy + matplotlib only; no Isaac):
    python3 eval_utils/visualize_camera_mounts.py
Outputs PNGs to isaac_images/ and a numeric summary to stdout.
"""

import json
import math
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(REPO, "sim_envs", "franka_orca_bimanual_cfg.py")
CALIB = os.path.join(REPO, "configs", "franka_orca_calibration.json")
OUT_DIR = os.path.join(REPO, "isaac_images")
os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# 1. Parse the live cfg (so this always reflects what the sim actually uses)
# ──────────────────────────────────────────────────────────────────────────
def _parse_cfg(path):
    src = open(path).read()

    def tup(name):
        m = re.search(rf"{name}\s*=\s*\(([^)]*)\)", src)
        return tuple(float(x) for x in m.group(1).split(",")[: None] if x.strip())

    eye_a = tup("_ARIA_EYE"); rot_a = tup("_ARIA_ROT")
    eye_o = tup("_OAKD_EYE"); rot_o = tup("_OAKD_ROT")
    arm_sep = float(re.search(r"ARM_SEPARATION_Y\s*=\s*([0-9.]+)", src).group(1))
    home = [float(re.search(rf'"panda_joint{i}":\s*(-?[0-9.]+)', src).group(1)) for i in range(1, 8)]

    # Lens params per camera block (NOT a global regex: the comments contain literals
    # like "focal_length=8 -> HFOV ~105 deg" that would poison a document-wide findall).
    def _block_val(block, key):
        for line in block.splitlines():
            s = line.strip()
            if s.startswith(f"{key}="):
                return float(re.search(rf"{key}=([0-9.]+)", s).group(1))
        return None

    aria_block = src[src.index("aria_rgb_cam = CameraCfg"):src.index("oakd_front_view = CameraCfg")]
    oakd_block = src[src.index("oakd_front_view = CameraCfg"):]
    focals = [_block_val(aria_block, "focal_length"), _block_val(oakd_block, "focal_length")]
    apers = [_block_val(aria_block, "horizontal_aperture"), _block_val(oakd_block, "horizontal_aperture")]
    return dict(eye_a=eye_a, rot_a=rot_a, eye_o=eye_o, rot_o=rot_o,
                arm_sep=arm_sep, home=home, focals=focals, apers=apers)


C = _parse_cfg(CFG)
ARM_SEP = C["arm_sep"]
LEFT_BASE = np.array([0.0, -ARM_SEP / 2, 0.0])
RIGHT_BASE = np.array([0.0, ARM_SEP / 2, 0.0])
HOME = C["home"]


# ──────────────────────────────────────────────────────────────────────────
# 2. Geometry helpers
# ──────────────────────────────────────────────────────────────────────────
def quat_to_R(q):
    """(w,x,y,z) -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def homogeneous(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Standard Franka Panda URDF joint frames (same as debug_camera_transforms.py)
PANDA = [
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
    return Rx @ Ry @ Rz


def franka_fk(q, base):
    T = homogeneous(np.eye(3), base)
    pts = [T[:3, 3].copy()]
    for jdef, qi in zip(PANDA, q):
        T = T @ homogeneous(_rpy(jdef["rpy"]), np.array(jdef["xyz"])) @ homogeneous(rot_z(qi), np.zeros(3))
        pts.append(T[:3, 3].copy())
    T = T @ homogeneous(np.eye(3), np.array([0, 0, 0.107]))  # flange
    pts.append(T[:3, 3].copy())
    return np.array(pts)


def hfov_vfov(focal, aper, w, h):
    hf = 2 * math.degrees(math.atan(aper / (2 * focal)))
    vap = aper * h / w
    vf = 2 * math.degrees(math.atan(vap / (2 * focal)))
    return hf, vf


# ──────────────────────────────────────────────────────────────────────────
# 3. Calibration ground truth (re-derive expected pose, compare to cfg)
# ──────────────────────────────────────────────────────────────────────────
def calib_expected():
    cal = json.load(open(CALIB))["camera_extrinsics_cam_to_base"]
    out = {}
    for tag, key, base in [("aria", "left_cam", LEFT_BASE), ("oakd", "right_cam", RIGHT_BASE)]:
        T = np.array(cal[key])
        t = T[:3, 3]
        pos = base + np.array([t[0], -t[1], t[2]])      # documented Y-flip
        fwd = T[:3, 2] / np.linalg.norm(T[:3, 2])       # OpenCV optical axis = +Z col
        out[tag] = (pos, fwd)
    return out


# ──────────────────────────────────────────────────────────────────────────
# 4. Drawing
# ──────────────────────────────────────────────────────────────────────────
def draw_axes(ax, o, R, scale=0.06, prefix=""):
    cols = ("#d62728", "#2ca02c", "#1f77b4")  # x r, y g, z b
    for i, c in enumerate(cols):
        e = o + scale * R[:, i]
        ax.plot(*[[o[k], e[k]] for k in range(3)], color=c, lw=2)


def draw_frustum(ax, o, R, hf, vf, color, length=0.28):
    """OpenGL frustum: camera looks down -Z (forward = -R[:,2])."""
    fwd = -R[:, 2]; right = R[:, 0]; up = R[:, 1]
    th = math.tan(math.radians(hf / 2)) * length
    tv = math.tan(math.radians(vf / 2)) * length
    c = o + fwd * length
    corners = [c + right * sx * th + up * sy * tv
               for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]]
    for cc in corners:
        ax.plot(*[[o[k], cc[k]] for k in range(3)], color=color, lw=0.9, alpha=0.7)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        ax.plot(*[[a[k], b[k]] for k in range(3)], color=color, lw=0.9, alpha=0.7)


def table_hit(o, fwd, z=0.05):
    if abs(fwd[2]) < 1e-6:
        return None
    t = (z - o[2]) / fwd[2]
    return o + t * fwd if t > 0 else None


def draw_table(ax):
    cx, cy = 0.6, 0.0
    hw, hd, zt = 0.45, 0.40, 0.05  # cfg table: size (0.9,0.8,0.05), centre x=0.6
    cs = np.array([[cx - hw, cy - hd, zt], [cx + hw, cy - hd, zt],
                   [cx + hw, cy + hd, zt], [cx - hw, cy + hd, zt]])
    for i in range(4):
        a, b = cs[i], cs[(i + 1) % 4]
        ax.plot(*[[a[k], b[k]] for k in range(3)], color="#8c564b", lw=1.5)


def draw_scene(ax, lpts, rpts, cams, elev, azim):
    draw_axes(ax, np.zeros(3), np.eye(3), scale=0.12)
    ax.text(0, 0, 0, " world", fontsize=6, color="k")
    for base, col, name in [(LEFT_BASE, "#1f77b4", "L_base"), (RIGHT_BASE, "#d62728", "R_base")]:
        ax.scatter(*base, color=col, s=55, zorder=5)
        ax.text(*base, f" {name}", fontsize=7, color=col)
    ax.plot(lpts[:, 0], lpts[:, 1], lpts[:, 2], "o-", color="#1f77b4", lw=2, ms=2.5, label="Left arm")
    ax.plot(rpts[:, 0], rpts[:, 1], rpts[:, 2], "o-", color="#d62728", lw=2, ms=2.5, label="Right arm")
    draw_table(ax)
    for cam in cams:
        o, R, hf, vf, col, name = cam["o"], cam["R"], cam["hf"], cam["vf"], cam["c"], cam["n"]
        ax.scatter(*o, color=col, s=70, marker="D", zorder=6)
        ax.text(o[0], o[1], o[2] + 0.02, f" {name}", fontsize=7, color=col, weight="bold")
        draw_axes(ax, o, R, scale=0.05)
        draw_frustum(ax, o, R, hf, vf, col)
        hit = table_hit(o, -R[:, 2])
        if hit is not None:
            ax.plot(*[[o[k], hit[k]] for k in range(3)], "--", color=col, lw=0.8, alpha=0.5)
            ax.scatter(*hit, color=col, s=25, marker="x", zorder=6)
    allp = np.vstack([lpts, rpts] + [c["o"] for c in cams] + [[0.6, 0, 0.05]])
    rng = (allp.max(0) - allp.min(0)).max() / 2 * 1.25
    mid = allp.mean(0)
    ax.set_xlim(mid[0] - rng, mid[0] + rng)
    ax.set_ylim(mid[1] - rng, mid[1] + rng)
    ax.set_zlim(0, mid[2] + rng)
    ax.set_xlabel("X fwd", fontsize=7); ax.set_ylabel("Y L/R", fontsize=7); ax.set_zlabel("Z up", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=elev, azim=azim)


# ──────────────────────────────────────────────────────────────────────────
# 5. Build + render
# ──────────────────────────────────────────────────────────────────────────
def main():
    lpts = franka_fk(HOME, LEFT_BASE)
    rpts = franka_fk(HOME, RIGHT_BASE)

    Ra = quat_to_R(C["rot_a"]); Ro = quat_to_R(C["rot_o"])
    hfa, vfa = hfov_vfov(C["focals"][0], C["apers"][0], 640, 480)
    hfo, vfo = hfov_vfov(C["focals"][1], C["apers"][1], 960, 540)
    cams = [
        dict(o=np.array(C["eye_a"]), R=Ra, hf=hfa, vf=vfa, c="#e377c2", n="aria"),
        dict(o=np.array(C["eye_o"]), R=Ro, hf=hfo, vf=vfo, c="#ff7f0e", n="oakd"),
    ]

    exp = calib_expected()

    # ---- numeric verdict ----
    print("=" * 74)
    print("CAMERA MOUNT CHECK  (cfg constants  vs  calibration ground truth)")
    print("=" * 74)
    for tag, cam in [("aria", cams[0]), ("oakd", cams[1])]:
        pos = cam["o"]; fwd = -cam["R"][:, 2]
        epos, efwd = exp[tag]
        dpos = pos - epos
        ang = math.degrees(math.acos(np.clip(np.dot(fwd, efwd), -1, 1)))
        pitch = math.degrees(math.asin(-fwd[2]))
        base = LEFT_BASE if tag == "aria" else RIGHT_BASE
        rel = pos - base
        print(f"\n{tag}:")
        print(f"  cfg world pos       = ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
        print(f"  calib expected pos  = ({epos[0]:+.3f}, {epos[1]:+.3f}, {epos[2]:+.3f})")
        print(f"  >> position residual= ({dpos[0]:+.4f}, {dpos[1]:+.4f}, {dpos[2]:+.4f})  |{np.linalg.norm(dpos):.4f}| m")
        print(f"  >> view-axis error  = {ang:.2f} deg   (cfg pitch {pitch:.1f} deg below horizontal)")
        print(f"  offset from {('L' if tag=='aria' else 'R')}_base = ({rel[0]:+.3f}, {rel[1]:+.3f}, {rel[2]:+.3f})")
        hit = table_hit(pos, fwd)
        if hit is not None:
            print(f"  optical axis hits table at (x={hit[0]:.3f}, y={hit[1]:.3f})")
    # camera midpoint vs arm-centre line (Y)
    midY = (cams[0]["o"][1] + cams[1]["o"][1]) / 2
    print(f"\n  arm centre Y = 0.000 ; camera mean Y = {midY:+.3f}  "
          f"(both cams between the bases? "
          f"{all(LEFT_BASE[1] < c['o'][1] < RIGHT_BASE[1] for c in cams)})")

    # ---- figure ----
    views = [(38, -60, "Perspective"), (89, -90, "Top-down"),
             (4, -90, "Side (from -Y)  — shows pitch/height"),
             (4, 0, "Front (from -X)  — shows L/R centring")]
    fig = plt.figure(figsize=(17, 14))
    fig.suptitle("Sim camera mounts vs Franka arms  (live cfg constants; ✕ = optical axis ∩ table)\n"
                 f"aria HFOV {hfa:.0f}°  oakd HFOV {hfo:.0f}°   "
                 f"residual vs calibration: aria {np.linalg.norm(cams[0]['o']-exp['aria'][0])*1000:.1f} mm, "
                 f"oakd {np.linalg.norm(cams[1]['o']-exp['oakd'][0])*1000:.1f} mm",
                 fontsize=13, fontweight="bold")
    for i, (elev, azim, name) in enumerate(views):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.set_title(name, fontsize=11)
        draw_scene(ax, lpts, rpts, cams, elev, azim)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "camera_mounts_current.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
