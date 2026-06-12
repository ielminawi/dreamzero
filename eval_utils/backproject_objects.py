"""Back-project GT pixel positions onto the table plane for a given camera.

Once the oakd camera is matched, run this to compute world (x, y) for each
annotated object pixel in the 960x540 GT frame, to place sim object stand-ins
where the training scene has them. Runs host-side (no GPU).
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=str, required=True,
                help='camera JSON: {"eye": [..], "dir": [..], "focal": 9.6}')
ap.add_argument("--points", type=str, required=True,
                help='JSON list: [{"name": "bag", "uv": [190, 270], "z": 0.05}, ...]')
args = ap.parse_args()

cam = json.loads(args.cam)
pts = json.loads(args.points)

eye = np.asarray(cam["eye"], float)
d = np.asarray(cam["dir"], float); d /= np.linalg.norm(d)
z_axis = -d
up = np.array([0.0, 0.0, 1.0])
x_axis = np.cross(up, z_axis); x_axis /= np.linalg.norm(x_axis)
y_axis = np.cross(z_axis, x_axis)
R = np.column_stack([x_axis, y_axis, z_axis])  # cam->world (OpenGL)

W, H, AP = 960, 540, 20.955
fx = cam["focal"] / AP * W

for p in pts:
    u, v = p["uv"]
    ray_cam = np.array([(u - W / 2) / fx, -(v - H / 2) / fx, -1.0])
    ray_w = R @ ray_cam
    t = (p.get("z", 0.05) - eye[2]) / ray_w[2]
    hit = eye + t * ray_w
    print(f"{p['name']:16s} uv=({u},{v}) -> world ({hit[0]:.3f}, {hit[1]:.3f}, {p.get('z', 0.05)})")
