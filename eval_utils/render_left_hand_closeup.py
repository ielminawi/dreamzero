"""Close-up multi-angle render of ONE hand at a recorded frame.

Teleports the arms to the xyzw arm-joint solution (from xyzw_compare.npz) and the
hands to recorded qpos_hand, reads the chosen hand's palm world pose, then orbits a
ring of close-up cameras around it (front/back/inner/outer/top/two iso) all aimed at
the palm, so the hand is framed from every side regardless of arm pose. Use it to
inspect finger closing direction after the left-mount fix.

Inside the Isaac container (docker/scripts/run_left_hand_closeup.sh):
  ... -p eval_utils/render_left_hand_closeup.py --headless --enable_cameras --device cpu \
      --h5 /app/20250826_111157.h5 --npz /app/xyzw_compare.npz --frame 1480 --side left \
      --out-dir /output
"""
import argparse
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--h5", type=str, default="/app/20250826_111157.h5")
parser.add_argument("--npz", type=str, required=True, help="xyzw_compare.npz (per-frame xyzw arm joints)")
parser.add_argument("--frame", type=int, default=1480)
parser.add_argument("--side", choices=["left", "right"], default="left")
parser.add_argument("--radius", type=float, default=0.22, help="camera distance from the palm (m)")
parser.add_argument("--out-dir", type=str, default="/output")
parser.add_argument("--warmup", type=int, default=40)
parser.add_argument("--res", type=int, default=1100, help="per-view square resolution")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os, sys
sys.path.insert(0, "/app")
import cv2
import omni.kit.app
_em = omni.kit.app.get_app().get_extension_manager()
for _ext in ["isaacsim.asset.importer.urdf", "omni.importer.urdf", "omni.isaac.urdf"]:
    if _em.set_extension_enabled_immediate(_ext, True):
        break

import h5py
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
import sim_envs  # noqa: F401
from sim_envs.franka_orca_bimanual_cfg import (
    FrankaOrcaBimanualEnvCfg, LEFT_ARM_ORDER, RIGHT_ARM_ORDER, LEFT_HAND_ORDER, RIGHT_HAND_ORDER)
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv

os.makedirs(args.out_dir, exist_ok=True)
def log(*a): print(" ".join(str(x) for x in a), flush=True)
def label(img, t, c=(255, 255, 255)):
    out = np.ascontiguousarray(img)
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, t, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)
    return out

# close-up view directions (unit offsets from the palm); cameras look AT the palm.
# keys must be USD-prim-safe (alnum/underscore); label is the human-readable axis.
VIEWS = [
    ("front",  "front +X", (1.0, 0.0, 0.15)),
    ("back",   "back -X",  (-1.0, 0.0, 0.15)),
    ("outer",  "outer -Y", (0.0, -1.0, 0.15)),   # left hand (+Y): looks from outside
    ("inner",  "inner +Y", (0.0, 1.0, 0.15)),
    ("top",    "top +Z",   (0.15, 0.0, 1.0)),
    ("iso_a",  "iso a",    (0.8, -0.8, 0.6)),
    ("iso_b",  "iso b",    (0.8, 0.8, 0.6)),
    ("iso_c",  "iso c",    (-0.8, -0.8, 0.6)),
]


def main():
    npz = np.load(args.npz, allow_pickle=True)
    frames = [int(x) for x in npz["frames"]]
    if args.frame not in frames:
        log(f"frame {args.frame} not in npz frames {frames}; using {frames[0]}"); args.frame = frames[0]
    fi = frames.index(args.frame)
    ikL = npz["ik_left"][fi, 0].astype(np.float32)   # cand 0 = xyzw
    ikR = npz["ik_right"][fi, 0].astype(np.float32)

    cfg = FrankaOrcaBimanualEnvCfg(); cfg.sim.device = args.device
    for name, _lbl, _ in VIEWS:
        setattr(cfg.scene, f"cu_{name}", CameraCfg(
            prim_path=f"/World/Closeup/{name}", update_period=0.0, height=args.res, width=args.res,
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world")))
    env = FrankaOrcaBimanualEnv(cfg); env.reset(); env.reset()

    with h5py.File(args.h5, "r") as f:
        q_hl = f["observations/qpos_hand_left"][args.frame].astype(np.float32)
        q_hr = f["observations/qpos_hand_right"][args.frame].astype(np.float32)

    idsL, _ = env.scene["left_arm_hand"].find_joints(list(LEFT_ARM_ORDER) + list(LEFT_HAND_ORDER), preserve_order=True)
    idsR, _ = env.scene["right_arm_hand"].find_joints(list(RIGHT_ARM_ORDER) + list(RIGHT_HAND_ORDER), preserve_order=True)
    for skey, ids, qa, qh in [("left_arm_hand", idsL, ikL, q_hl), ("right_arm_hand", idsR, ikR, q_hr)]:
        art = env.scene[skey]
        pos = torch.tensor(np.concatenate([qa, qh]), dtype=torch.float32, device=env.device).unsqueeze(0)
        art.write_joint_state_to_sim(pos, torch.zeros_like(pos), joint_ids=ids)
        art.set_joint_position_target(pos, joint_ids=ids)
    env.scene.write_data_to_sim()

    a = torch.zeros((1, 48), dtype=torch.float32, device=env.device)
    a[0, 0:7] = torch.tensor(ikL, device=env.device); a[0, 7:14] = torch.tensor(ikR, device=env.device)
    a[0, 14:31] = torch.tensor(q_hl, device=env.device); a[0, 31:48] = torch.tensor(q_hr, device=env.device)
    for _ in range(5):
        env.step(a)

    # palm world position of the requested hand
    art = env.scene[f"{args.side}_arm_hand"]
    bid, _ = art.find_bodies(f"{args.side}_palm")
    C = art.data.body_state_w[0, bid[0], 0:3].cpu().numpy()
    log(f"{args.side}_palm world pos = {np.round(C, 3)}; aiming {len(VIEWS)} close-ups at radius {args.radius}")

    for name, _lbl, d in VIEWS:
        d = np.array(d, float); d = d / np.linalg.norm(d)
        eye = C + args.radius * d
        env.scene[f"cu_{name}"].set_world_poses_from_view(
            torch.tensor([eye], dtype=torch.float32, device=env.device),
            torch.tensor([C], dtype=torch.float32, device=env.device))
    for _ in range(args.warmup):
        env.step(a)

    tiles = []
    for name, lbl, _ in VIEWS:
        img = env.scene[f"cu_{name}"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(args.out_dir, f"{args.side}_closeup_{name}_t{args.frame}.png"),
                    cv2.cvtColor(label(img, f"{args.side} {lbl} t{args.frame}", (120, 255, 120)), cv2.COLOR_RGB2BGR))
        tiles.append(label(img, f"{args.side} {lbl}", (120, 255, 120)))
    # montage 4 x 2
    TS = 540
    tiles = [cv2.resize(t, (TS, TS)) for t in tiles]
    rows = [np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, len(tiles), 4)]
    montage = np.concatenate(rows, axis=0)
    outp = os.path.join(args.out_dir, f"{args.side}_closeup_montage_t{args.frame}.png")
    cv2.imwrite(outp, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    log("wrote", outp); log("DONE"); env.close(); simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("\n*** EXCEPTION ***"); log(traceback.format_exc()); os._exit(1)
