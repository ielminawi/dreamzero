#!/usr/bin/env python3
"""Debug script: visualise camera/robot transforms and test hypotheses.

Shows 3 side-by-side scenarios:
  A) CURRENT  — cameras offset from each arm's base (cameras end up outside)
  B) HYPOTHESIS 1 — "base" = world origin (no arm offset, use calib positions directly)
  C) HYPOTHESIS 2 — "base" = each arm's base, but negate Y offset

Run:  python3 debug_camera_transforms.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation

# ──────────────────────────────────────────────
# 1. Constants
# ──────────────────────────────────────────────
ARM_SEPARATION_Y = 0.6127
LEFT_ARM_BASE  = np.array([0.0, -ARM_SEPARATION_Y / 2, 0.0])
RIGHT_ARM_BASE = np.array([0.0,  ARM_SEPARATION_Y / 2, 0.0])
HOME_JOINTS = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]

# ──────────────────────────────────────────────
# 2. Camera calibration (cam -> base)
# ──────────────────────────────────────────────
calib_left = np.array([
    [-0.02199727, -0.80581615,  0.59175708,  0.20403467],
    [-0.99905014,  0.03998766,  0.01731508, -0.25486327],
    [-0.03761575, -0.59081411, -0.80593036,  0.43379187],
    [ 0.0,         0.0,         0.0,         1.0        ],
])
calib_right = np.array([
    [ 0.02933941, -0.83227828,  0.55358113,  0.17515134],
    [-0.99642232,  0.01956109,  0.0822187,   0.34649483],
    [-0.07925749, -0.55401284, -0.82872675,  0.46895363],
    [ 0.0,         0.0,         0.0,         1.0        ],
])

# ──────────────────────────────────────────────
# 3. Franka FK
# ──────────────────────────────────────────────
def homogeneous(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

PANDA_JOINTS_URDF = [
    {"xyz": [0, 0, 0.333],       "rpy": [0, 0, 0]},
    {"xyz": [0, 0, 0],           "rpy": [-np.pi/2, 0, 0]},
    {"xyz": [0, -0.316, 0],      "rpy": [np.pi/2, 0, 0]},
    {"xyz": [0.0825, 0, 0],      "rpy": [np.pi/2, 0, 0]},
    {"xyz": [-0.0825, 0.384, 0], "rpy": [-np.pi/2, 0, 0]},
    {"xyz": [0, 0, 0],           "rpy": [np.pi/2, 0, 0]},
    {"xyz": [0.088, 0, 0],       "rpy": [np.pi/2, 0, 0]},
]

def franka_fk(joint_angles, base_pos=np.zeros(3)):
    T = homogeneous(np.eye(3), base_pos)
    positions = [T[:3, 3].copy()]
    for jdef, q in zip(PANDA_JOINTS_URDF, joint_angles):
        R_fixed = Rotation.from_euler("xyz", jdef["rpy"]).as_matrix()
        T = T @ homogeneous(R_fixed, np.array(jdef["xyz"])) @ homogeneous(rot_z(q), np.zeros(3))
        positions.append(T[:3, 3].copy())
    T = T @ homogeneous(np.eye(3), np.array([0, 0, 0.107]))
    positions.append(T[:3, 3].copy())
    return positions, T

# ──────────────────────────────────────────────
# 4. Drawing helpers
# ──────────────────────────────────────────────
def draw_frame(ax, origin, R, scale=0.1, labels=("X","Y","Z"),
               colors=("r","g","b"), label_prefix="", lw=2, fontsize=7):
    for i, (col, lbl) in enumerate(zip(colors, labels)):
        end = origin + scale * R[:, i]
        ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]],
                color=col, linewidth=lw)
        ax.text(end[0], end[1], end[2], f"{label_prefix}{lbl}", fontsize=fontsize, color=col)

def draw_frustum(ax, origin, R_cv, color="cyan", length=0.15, fov_deg=50):
    ha = np.radians(fov_deg / 2); d = length; h = d * np.tan(ha)
    corners = np.array([[h,h,d],[-h,h,d],[-h,-h,d],[h,-h,d]])
    cw = (R_cv @ corners.T).T + origin
    for c in cw:
        ax.plot([origin[0],c[0]], [origin[1],c[1]], [origin[2],c[2]],
                color=color, linewidth=0.8, alpha=0.6)
    for i in range(4):
        j = (i+1) % 4
        ax.plot([cw[i,0],cw[j,0]], [cw[i,1],cw[j,1]], [cw[i,2],cw[j,2]],
                color=color, linewidth=0.8, alpha=0.6)

def draw_table(ax):
    tc = np.array([0.5, 0.0, 0.0])
    tw, td, th = 0.6, 0.4, 0.025
    corners = np.array([
        [tc[0]-tw, tc[1]-td, tc[2]+th], [tc[0]+tw, tc[1]-td, tc[2]+th],
        [tc[0]+tw, tc[1]+td, tc[2]+th], [tc[0]-tw, tc[1]+td, tc[2]+th],
    ])
    for i in range(4):
        j = (i+1) % 4
        ax.plot([corners[i,0],corners[j,0]], [corners[i,1],corners[j,1]],
                [corners[i,2],corners[j,2]], color="brown", linewidth=2)
    ax.text(0.5, 0, 0.05, "Table", fontsize=7, color="brown")
    return corners

def draw_scene_common(ax, left_pts, right_pts):
    """Draw arms, bases, table, world frame."""
    draw_frame(ax, np.zeros(3), np.eye(3), scale=0.12, label_prefix="W_", fontsize=6)

    ax.scatter(*LEFT_ARM_BASE, color="blue", s=60, zorder=5)
    ax.text(*LEFT_ARM_BASE, " L_base", fontsize=7, color="blue")
    ax.scatter(*RIGHT_ARM_BASE, color="red", s=60, zorder=5)
    ax.text(*RIGHT_ARM_BASE, " R_base", fontsize=7, color="red")

    ax.plot(left_pts[:,0], left_pts[:,1], left_pts[:,2],
            "o-", color="blue", linewidth=2, markersize=3, label="Left arm")
    ax.plot(right_pts[:,0], right_pts[:,1], right_pts[:,2],
            "o-", color="red", linewidth=2, markersize=3, label="Right arm")
    ax.scatter(*left_pts[-1], color="blue", s=60, marker="^", zorder=5)
    ax.scatter(*right_pts[-1], color="red", s=60, marker="^", zorder=5)

    draw_table(ax)

def set_equal_axes(ax, all_pts, elev=50, azim=-60):
    max_range = (all_pts.max(0) - all_pts.min(0)).max() / 2 * 1.3
    mid = all_pts.mean(0)
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
    ax.set_xlabel("X (fwd)", fontsize=7)
    ax.set_ylabel("Y (L/R)", fontsize=7)
    ax.set_zlabel("Z (up)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=elev, azim=azim)

# ──────────────────────────────────────────────
# 5. Compute FK
# ──────────────────────────────────────────────
left_fk, _ = franka_fk(HOME_JOINTS, LEFT_ARM_BASE)
right_fk, _ = franka_fk(HOME_JOINTS, RIGHT_ARM_BASE)
left_pts = np.array(left_fk)
right_pts = np.array(right_fk)
all_pts = np.vstack([left_pts, right_pts, [[0,0,0]]])

# ──────────────────────────────────────────────
# 6. Three hypotheses for camera placement
# ──────────────────────────────────────────────
R_left_cv = calib_left[:3, :3]
R_right_cv = calib_right[:3, :3]
t_left = calib_left[:3, 3]
t_right = calib_right[:3, 3]

hypotheses = {
    "A) CURRENT\n(base = per-arm, pos = arm + calib_t)": {
        "aria_pos":  LEFT_ARM_BASE + t_left,
        "oakd_pos":  RIGHT_ARM_BASE + t_right,
        "aria_R_cv": R_left_cv,
        "oakd_R_cv": R_right_cv,
    },
    "B) base = world origin\n(pos = calib_t directly)": {
        "aria_pos":  t_left.copy(),
        "oakd_pos":  t_right.copy(),
        "aria_R_cv": R_left_cv,
        "oakd_R_cv": R_right_cv,
    },
    "C) base = per-arm, negate Y\n(pos = arm + [tx, -ty, tz])": {
        "aria_pos":  LEFT_ARM_BASE + np.array([t_left[0], -t_left[1], t_left[2]]),
        "oakd_pos":  RIGHT_ARM_BASE + np.array([t_right[0], -t_right[1], t_right[2]]),
        "aria_R_cv": R_left_cv,
        "oakd_R_cv": R_right_cv,
    },
}

# ──────────────────────────────────────────────
# 7. Big comparison figure — 3 columns x 2 rows
#    Row 1: perspective view, Row 2: top-down
# ──────────────────────────────────────────────
fig = plt.figure(figsize=(24, 16))
fig.suptitle("Camera Placement Hypotheses — which matches your real system?",
             fontsize=14, fontweight="bold")

for col_idx, (title, hyp) in enumerate(hypotheses.items()):
    aria_pos = hyp["aria_pos"]
    oakd_pos = hyp["oakd_pos"]
    aria_R = hyp["aria_R_cv"]
    oakd_R = hyp["oakd_R_cv"]

    # ---- Row 1: perspective ----
    ax = fig.add_subplot(2, 3, col_idx + 1, projection="3d")
    ax.set_title(title, fontsize=10, fontweight="bold")
    draw_scene_common(ax, left_pts, right_pts)

    # Aria camera
    draw_frame(ax, aria_pos, aria_R, scale=0.06, labels=("x","y","z"), label_prefix="Aria_", fontsize=5)
    draw_frustum(ax, aria_pos, aria_R, color="magenta", length=0.18)
    ax.scatter(*aria_pos, color="magenta", s=80, marker="D", zorder=5)
    ax.text(*aria_pos, f"  Aria\n  Y={aria_pos[1]:.3f}", fontsize=6, color="magenta")

    # OakD camera
    draw_frame(ax, oakd_pos, oakd_R, scale=0.06, labels=("x","y","z"), label_prefix="OakD_", fontsize=5)
    draw_frustum(ax, oakd_pos, oakd_R, color="orange", length=0.18)
    ax.scatter(*oakd_pos, color="orange", s=80, marker="D", zorder=5)
    ax.text(*oakd_pos, f"  OakD\n  Y={oakd_pos[1]:.3f}", fontsize=6, color="orange")

    pts = np.vstack([all_pts, [aria_pos], [oakd_pos]])
    set_equal_axes(ax, pts, elev=45, azim=-55)
    ax.legend(fontsize=6, loc="upper left")

    # ---- Row 2: top-down ----
    ax2 = fig.add_subplot(2, 3, col_idx + 4, projection="3d")
    ax2.set_title(f"{title}\n(top-down)", fontsize=9)
    draw_scene_common(ax2, left_pts, right_pts)

    draw_frustum(ax2, aria_pos, aria_R, color="magenta", length=0.22)
    ax2.scatter(*aria_pos, color="magenta", s=80, marker="D", zorder=5)
    ax2.text(aria_pos[0], aria_pos[1], aria_pos[2],
             f" Aria Y={aria_pos[1]:.3f}", fontsize=7, color="magenta")

    draw_frustum(ax2, oakd_pos, oakd_R, color="orange", length=0.22)
    ax2.scatter(*oakd_pos, color="orange", s=80, marker="D", zorder=5)
    ax2.text(oakd_pos[0], oakd_pos[1], oakd_pos[2],
             f" OakD Y={oakd_pos[1]:.3f}", fontsize=7, color="orange")

    # Draw vertical lines to show Y relationship to arm bases
    for cam_p, cam_c in [(aria_pos, "magenta"), (oakd_pos, "orange")]:
        ax2.plot([cam_p[0], cam_p[0]], [cam_p[1], cam_p[1]], [0, cam_p[2]],
                 "--", color=cam_c, alpha=0.3, linewidth=1)

    set_equal_axes(ax2, pts, elev=90, azim=-90)

plt.tight_layout()
out1 = "/Users/noasendlhofer/Documents/GitHub/dreamzero/isaac_images/camera_hypotheses.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ──────────────────────────────────────────────
# 8. Print numeric summary for each hypothesis
# ──────────────────────────────────────────────
print("\n" + "="*70)
print("NUMERIC SUMMARY")
print("="*70)
for title, hyp in hypotheses.items():
    aria_pos = hyp["aria_pos"]
    oakd_pos = hyp["oakd_pos"]
    print(f"\n{title}")
    print(f"  Aria pos:  ({aria_pos[0]:.3f}, {aria_pos[1]:.3f}, {aria_pos[2]:.3f})")
    print(f"  OakD pos:  ({oakd_pos[0]:.3f}, {oakd_pos[1]:.3f}, {oakd_pos[2]:.3f})")
    print(f"  Left arm base Y  = {LEFT_ARM_BASE[1]:.3f}")
    print(f"  Right arm base Y = {RIGHT_ARM_BASE[1]:.3f}")
    aria_between = LEFT_ARM_BASE[1] < aria_pos[1] < RIGHT_ARM_BASE[1]
    oakd_between = LEFT_ARM_BASE[1] < oakd_pos[1] < RIGHT_ARM_BASE[1]
    print(f"  Aria between arms? {aria_between}  (Y in [{LEFT_ARM_BASE[1]:.3f}, {RIGHT_ARM_BASE[1]:.3f}])")
    print(f"  OakD between arms? {oakd_between}  (Y in [{LEFT_ARM_BASE[1]:.3f}, {RIGHT_ARM_BASE[1]:.3f}])")

    # Camera look direction
    aria_look = hyp["aria_R_cv"][:, 2]
    oakd_look = hyp["oakd_R_cv"][:, 2]
    print(f"  Aria looks toward:  ({aria_look[0]:+.3f}, {aria_look[1]:+.3f}, {aria_look[2]:+.3f})")
    print(f"  OakD looks toward:  ({oakd_look[0]:+.3f}, {oakd_look[1]:+.3f}, {oakd_look[2]:+.3f})")
    print(f"    (both look forward+down — this is the same for all hypotheses)")

# ──────────────────────────────────────────────
# 9. Additional detail figure: just hypothesis B
#    with labelled distances, multiple view angles
# ──────────────────────────────────────────────
fig2 = plt.figure(figsize=(20, 10))
fig2.suptitle("Hypothesis B detail (base = world origin)", fontsize=13, fontweight="bold")

hyp_b = hypotheses["B) base = world origin\n(pos = calib_t directly)"]
aria_pos = hyp_b["aria_pos"]
oakd_pos = hyp_b["oakd_pos"]

for subplot_idx, (elev, azim, view_name) in enumerate([
    (45, -55, "Perspective"), (90, -90, "Top-down"),
    (0, -90, "Side (from -Y)"), (0, 0, "Front (from -X)")
]):
    ax = fig2.add_subplot(1, 4, subplot_idx + 1, projection="3d")
    ax.set_title(f"{view_name}", fontsize=9)
    draw_scene_common(ax, left_pts, right_pts)

    draw_frame(ax, aria_pos, R_left_cv, scale=0.06, labels=("x","y","z"),
               label_prefix="A_", fontsize=5)
    draw_frustum(ax, aria_pos, R_left_cv, color="magenta", length=0.2)
    ax.scatter(*aria_pos, color="magenta", s=80, marker="D", zorder=5)
    ax.text(*aria_pos, " Aria", fontsize=7, color="magenta")

    draw_frame(ax, oakd_pos, R_right_cv, scale=0.06, labels=("x","y","z"),
               label_prefix="O_", fontsize=5)
    draw_frustum(ax, oakd_pos, R_right_cv, color="orange", length=0.2)
    ax.scatter(*oakd_pos, color="orange", s=80, marker="D", zorder=5)
    ax.text(*oakd_pos, " OakD", fontsize=7, color="orange")

    pts = np.vstack([all_pts, [aria_pos], [oakd_pos]])
    set_equal_axes(ax, pts, elev=elev, azim=azim)

plt.tight_layout()
out2 = "/Users/noasendlhofer/Documents/GitHub/dreamzero/isaac_images/camera_hypothesis_B_detail.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# Also do hypothesis C detail
fig3 = plt.figure(figsize=(20, 10))
fig3.suptitle("Hypothesis C detail (base = per-arm, negate Y)", fontsize=13, fontweight="bold")

hyp_c = hypotheses["C) base = per-arm, negate Y\n(pos = arm + [tx, -ty, tz])"]
aria_pos_c = hyp_c["aria_pos"]
oakd_pos_c = hyp_c["oakd_pos"]

for subplot_idx, (elev, azim, view_name) in enumerate([
    (45, -55, "Perspective"), (90, -90, "Top-down"),
    (0, -90, "Side (from -Y)"), (0, 0, "Front (from -X)")
]):
    ax = fig3.add_subplot(1, 4, subplot_idx + 1, projection="3d")
    ax.set_title(f"{view_name}", fontsize=9)
    draw_scene_common(ax, left_pts, right_pts)

    draw_frame(ax, aria_pos_c, R_left_cv, scale=0.06, labels=("x","y","z"),
               label_prefix="A_", fontsize=5)
    draw_frustum(ax, aria_pos_c, R_left_cv, color="magenta", length=0.2)
    ax.scatter(*aria_pos_c, color="magenta", s=80, marker="D", zorder=5)
    ax.text(*aria_pos_c, " Aria", fontsize=7, color="magenta")

    draw_frame(ax, oakd_pos_c, R_right_cv, scale=0.06, labels=("x","y","z"),
               label_prefix="O_", fontsize=5)
    draw_frustum(ax, oakd_pos_c, R_right_cv, color="orange", length=0.2)
    ax.scatter(*oakd_pos_c, color="orange", s=80, marker="D", zorder=5)
    ax.text(*oakd_pos_c, " OakD", fontsize=7, color="orange")

    pts = np.vstack([all_pts, [aria_pos_c], [oakd_pos_c]])
    set_equal_axes(ax, pts, elev=elev, azim=azim)

plt.tight_layout()
out3 = "/Users/noasendlhofer/Documents/GitHub/dreamzero/isaac_images/camera_hypothesis_C_detail.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")
