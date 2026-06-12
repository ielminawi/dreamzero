"""Freeze the bimanual rig at one or more recorded frames and render each from
many angles at high resolution, to diagnose arm-mounting / orientation problems.

Motivation: the real-vs-sim replay shows the arms "shifted" — possibly the bases are
oriented wrong (e.g. a 45 deg yaw), or the joint pose is mismapped. The two on-robot
cameras alone can't disambiguate a base-yaw from a joint problem, so this script ALSO
drops free overview cameras (front, top-down, side, perspective) that see the whole
rig from outside, plus it pulls the matching REAL aria/oak-d frames straight from the
h5 for a direct sim-vs-real comparison at the SAME instant.

What it does:
  1. Build the FrankaOrcaBimanual env ONCE, with 4 extra high-res overview cameras.
  2. For each requested frame F: teleport BOTH arm+hand articulations to the recorded
     state (qpos[F]), with the same real->sim arm remap the replay uses, HOLD that pose
     (target=qpos), settle a few steps, and capture all cameras.
  3. Save high-res PNGs for every view + a labelled montage per frame, real|sim
     comparisons for the two robot cameras, and ONE cross-frame contact sheet
     (a row per frame: persp | real aria | sim aria | real oakd | sim oakd).

Outputs (per frame, to --out-dir/frame<F>_views):
  sim_aria.png  real_aria.png  cmp_aria.png        (640x480 robot cam, real|sim)
  sim_oakd.png  real_oakd.png  cmp_oakd.png        (960x540 robot cam, real|sim)
  overview_front.png  overview_top.png  overview_side.png  overview_persp.png  (hi-res)
  montage_frame<F>.png                              (everything, labelled, on one canvas)
Plus (to --out-dir):
  contact_sheet.png                                 (one row per frame)

Run inside the Isaac container (see docker/scripts/run_snapshot_frame0.sh), e.g.:
  /opt/IsaacLab/isaaclab.sh -p eval_utils/snapshot_frame0_multiview.py \
      --headless --enable_cameras --device cpu --h5 /app/20250826_111157.h5 \
      --frames 0,400,800,1200
"""

import argparse

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--h5", type=str, default="/app/20250826_111157.h5")
parser.add_argument("--out-dir", type=str, default="/output")
parser.add_argument("--frames", type=str, default="0",
                    help="comma-separated recorded frame indices to freeze at")
parser.add_argument("--frame", type=int, default=None,
                    help="(back-compat) single frame; overrides --frames if given")
parser.add_argument("--warmup", type=int, default=30,
                    help="settle/camera-warmup steps before the FIRST capture")
parser.add_argument("--settle", type=int, default=20,
                    help="settle steps after each subsequent teleport")
parser.add_argument("--ov-w", type=int, default=1920, help="overview camera width (hi-res)")
parser.add_argument("--ov-h", type=int, default=1440, help="overview camera height (hi-res)")
parser.add_argument("--remap-arm", dest="remap_arm", action="store_true", default=True,
                    help="apply configured real->sim arm joint-convention remap")
parser.add_argument("--no-remap-arm", dest="remap_arm", action="store_false")
parser.add_argument("--joints-npz", type=str, default=None,
                    help="exact IK arm-joint trajectories (export_ik_joints.py: ik_left/ik_right). "
                         "If given, the ARM is posed from these per-frame solutions (the EE-pose "
                         "interpretation under test) instead of the real->sim joint-angle remap; "
                         "hands still use recorded qpos_hand.")
parser.add_argument("--no-objects", action="store_true",
                    help="remove the table objects (grocery bag, container, tube, round item) "
                         "so the hands are unobstructed in the renders")
parser.add_argument("--neutral-hands", action="store_true",
                    help="pose all 17 hand joints at 0 (flat/neutral) instead of the recorded "
                         "qpos_hand of the frame (for presentation renders)")
parser.add_argument("--dark-bg", action="store_true",
                    help="completely dark background: remove the grid ground plane and the "
                         "room backdrop (floor + back wall); table and lights stay")
parser.add_argument("--table-under-arms", action="store_true",
                    help="extend the table VISUALLY under the arm bases (visual-only cuboid, "
                         "no collision: a collidable top overlapping the bases hangs PhysX)")
parser.add_argument("--table-width", type=float, default=None,
                    help="override the table width (y extent, meters; default 0.8). Also "
                         "applies to the --table-under-arms extension.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys
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

import sim_envs  # noqa: F401  (registers the env)
from sim_envs.franka_orca_bimanual_cfg import (
    FrankaOrcaBimanualEnvCfg, real_arm_to_sim, ARM_SIM_FROM_REAL,
    LEFT_ARM_ORDER, RIGHT_ARM_ORDER, LEFT_HAND_ORDER, RIGHT_HAND_ORDER,
)
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv

ARIA_DS = "observations/images/aria_rgb_cam/color"
OAKD_DS = "observations/images/oakd_front_view/color"

FRAMES = [args.frame] if args.frame is not None else [int(t) for t in args.frames.split(",")]
os.makedirs(args.out_dir, exist_ok=True)


def log(*a):
    print(" ".join(str(x) for x in a), flush=True)


def label(img, text, color=(255, 255, 255), scale=0.8):
    out = np.ascontiguousarray(img)
    h = int(34 * scale / 0.8)
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (8, int(h * 0.78)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return out


def save_rgb(path, rgb, text=None, color=(255, 255, 255)):
    img = rgb[..., :3].astype(np.uint8)
    if text is not None:
        img = label(img, text, color)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img


def fit(img, w, h):
    return cv2.resize(img[..., :3].astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA)


# Overview cameras: (name, eye, target, focal_mm). World frame. Arm bases at x=0,
# y=+/-0.306; table top z=0.05, workspace ~ (0.35, 0, 0.1). The visual backdrop wall sits
# at x=1.5, so the +X-side cameras (front/persp) MUST stay in front of it (x<1.45) or they
# just film the wall. Slight off-axis on the top view avoids the look-at degeneracy
# (forward antiparallel to world-up) and reads depth better.
OVERVIEWS = [
    ("front", (1.42, 0.0, 0.70), (0.10, 0.0, 0.20), 9.0),    # +X looking back -X: both arms L|R (wide; wall at x=1.5)
    ("top",   (0.30, 0.0, 2.20), (0.38, 0.0, 0.05), 26.0),   # near-straight-down: reveals base YAW
    ("side",  (0.38, -1.45, 0.45), (0.38, 0.0, 0.18), 14.0),  # -Y looking +Y: reach + height profile
    ("persp", (1.15, -1.15, 0.95), (0.28, 0.0, 0.12), 14.0),  # iso
]


def capture_frame(env, f, F, out_dir):
    """Teleport to recorded frame F, settle, capture all cameras, save outputs.

    Returns the contact-sheet row (RGB)."""
    q_al = f["observations/qpos_arm_left"][F].astype(np.float32)
    q_ar = f["observations/qpos_arm_right"][F].astype(np.float32)
    q_hl = f["observations/qpos_hand_left"][F].astype(np.float32)
    q_hr = f["observations/qpos_hand_right"][F].astype(np.float32)
    if args.neutral_hands:
        q_hl = np.zeros(17, dtype=np.float32)
        q_hr = np.zeros(17, dtype=np.float32)
    real_aria = f[ARIA_DS][F].astype(np.uint8)   # (480,640,3) RGB
    real_oakd = f[OAKD_DS][F].astype(np.uint8)   # (540,960,3) RGB
    log(f"frame-{F} qpos_arm_left  (real) = {np.array2string(q_al, precision=3)}")
    log(f"frame-{F} qpos_arm_right (real) = {np.array2string(q_ar, precision=3)}")
    if capture_frame.jq is not None:
        jql, jqr = capture_frame.jq
        q_al = jql[F].astype(np.float32)
        q_ar = jqr[F].astype(np.float32)
        log(f"frame-{F} arm joints from npz (exact IK) L={np.array2string(q_al, precision=3)}")
        log(f"frame-{F} arm joints from npz (exact IK) R={np.array2string(q_ar, precision=3)}")
    elif args.remap_arm:
        q_al = real_arm_to_sim(q_al, "left").astype(np.float32)
        q_ar = real_arm_to_sim(q_ar, "right").astype(np.float32)
        log(f"frame-{F} qpos_arm_left  (sim)  = {np.array2string(q_al, precision=3)}")
        log(f"frame-{F} qpos_arm_right (sim)  = {np.array2string(q_ar, precision=3)}")

    OUT = os.path.join(out_dir, f"frame{F}_views")
    os.makedirs(OUT, exist_ok=True)

    # ---- freeze both arms at the recorded frame-F state, hold target=qpos ----
    for skey, side, qarm, qhand in [
        ("left_arm_hand", "left", q_al, q_hl),
        ("right_arm_hand", "right", q_ar, q_hr),
    ]:
        art = env.scene[skey]
        names = (LEFT_ARM_ORDER + LEFT_HAND_ORDER) if side == "left" else (RIGHT_ARM_ORDER + RIGHT_HAND_ORDER)
        ids, _ = art.find_joints(list(names), preserve_order=True)
        vals = np.concatenate([qarm, qhand]).astype(np.float32)
        pos = torch.tensor(vals, device=env.device).unsqueeze(0)
        art.write_joint_state_to_sim(pos, torch.zeros_like(pos), joint_ids=ids)
        art.set_joint_position_target(pos, joint_ids=ids)
    env.scene.write_data_to_sim()

    # ---- settle / hydrate camera buffers ----
    a_hold = torch.zeros((1, 48), dtype=torch.float32, device=env.device)
    a_hold[0, 0:7] = torch.tensor(q_al, device=env.device)
    a_hold[0, 7:14] = torch.tensor(q_ar, device=env.device)
    a_hold[0, 14:31] = torch.tensor(q_hl, device=env.device)
    a_hold[0, 31:48] = torch.tensor(q_hr, device=env.device)
    steps = args.warmup if capture_frame.first else args.settle
    capture_frame.first = False
    for _ in range(max(1, steps)):
        env.step(a_hold)

    # ---- read sim cameras ----
    sim_aria = env.scene["aria_rgb_cam"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    sim_oakd = env.scene["oakd_front_view"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    ov_imgs = {}
    for name, *_ in OVERVIEWS:
        ov_imgs[name] = env.scene[f"ov_{name}"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)

    # ---- save individual PNGs ----
    save_rgb(os.path.join(OUT, "sim_aria.png"), sim_aria, f"SIM aria  frame {F}", (120, 200, 255))
    save_rgb(os.path.join(OUT, "real_aria.png"), real_aria, f"REAL aria  frame {F}", (120, 255, 120))
    save_rgb(os.path.join(OUT, "sim_oakd.png"), sim_oakd, f"SIM oak-d  frame {F}", (120, 200, 255))
    save_rgb(os.path.join(OUT, "real_oakd.png"), real_oakd, f"REAL oak-d  frame {F}", (120, 255, 120))
    for name, *_ in OVERVIEWS:
        save_rgb(os.path.join(OUT, f"overview_{name}.png"), ov_imgs[name],
                 f"OVERVIEW {name}  frame {F}", (255, 230, 120))

    # ---- real|sim comparisons for the robot cameras ----
    def cmp(real, sim, nm):
        r = label(real.copy(), f"REAL {nm} f{F}", (120, 255, 120))
        s = label(sim.copy(), f"SIM {nm} f{F}", (120, 200, 255))
        out = np.concatenate([r, s], axis=1)
        cv2.imwrite(os.path.join(OUT, f"cmp_{nm}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    cmp(real_aria, sim_aria, "aria")
    cmp(real_oakd, sim_oakd, "oakd")

    # ---- montage: overviews (top row) + robot-cam real|sim (bottom rows) ----
    TW = 640  # montage tile width
    row1 = np.concatenate([
        label(fit(ov_imgs[n], TW, int(TW * args.ov_h / args.ov_w)), n, (255, 230, 120))
        for n, *_ in OVERVIEWS], axis=1)

    def pair_row(real, sim, nm):
        r = label(fit(real, TW, 480), f"REAL {nm}", (120, 255, 120))
        s = label(fit(sim, TW, 480), f"SIM {nm}", (120, 200, 255))
        pair = np.concatenate([r, s], axis=1)
        return cv2.resize(pair, (row1.shape[1], int(pair.shape[0] * row1.shape[1] / pair.shape[1])))
    row2 = pair_row(real_aria, sim_aria, "aria")
    row3 = pair_row(real_oakd, sim_oakd, "oakd")
    montage = np.concatenate([row1, row2, row3], axis=0)
    cv2.imwrite(os.path.join(OUT, f"montage_frame{F}.png"), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))

    # ---- contact-sheet row: persp | real aria | sim aria | real oakd | sim oakd ----
    RH = 360
    row = np.concatenate([
        label(fit(ov_imgs["persp"], 480, RH), f"persp f{F}", (255, 230, 120), 0.6),
        label(fit(real_aria, 480, RH), f"REAL aria f{F}", (120, 255, 120), 0.6),
        label(fit(sim_aria, 480, RH), f"SIM aria f{F}", (120, 200, 255), 0.6),
        label(fit(real_oakd, 640, RH), f"REAL oakd f{F}", (120, 255, 120), 0.6),
        label(fit(sim_oakd, 640, RH), f"SIM oakd f{F}", (120, 200, 255), 0.6),
    ], axis=1)
    log(f"captured frame {F} -> {OUT}")
    return row


capture_frame.first = True
capture_frame.jq = None


def main():
    log("=" * 72)
    log(f"MULTIVIEW SNAPSHOT  h5={args.h5}  frames={FRAMES}  remap_arm={args.remap_arm}")
    log("=" * 72)

    # ---- build cfg and INJECT overview cameras before constructing the env ----
    cfg = FrankaOrcaBimanualEnvCfg()
    cfg.sim.device = args.device
    if args.no_objects:
        for obj in ["grocery_bag", "white_container", "blue_tube", "round_item"]:
            if hasattr(cfg.scene, obj):
                setattr(cfg.scene, obj, None)
        log("removed table objects from the scene")
    if args.dark_bg:
        for obj in ["ground", "floor_visual", "back_wall"]:
            if hasattr(cfg.scene, obj):
                setattr(cfg.scene, obj, None)
        log("dark background: removed ground grid + room backdrop")
    tw = args.table_width if args.table_width else 0.8
    if args.table_width:
        s = cfg.scene.table.spawn.size
        cfg.scene.table.spawn.size = (s[0], tw, s[2])
        log(f"table width override: {tw} m")
    if args.table_under_arms:
        from isaaclab.assets import AssetBaseCfg
        # matches the real table (size/color of cfg.scene.table), spanning x [-0.35, 0.15]
        # so it sits under the arm bases at x=0; top flush with the real table at z=0.05
        cfg.scene.table_ext = AssetBaseCfg(
            prim_path="/World/TableExt",
            spawn=sim_utils.CuboidCfg(
                size=(0.5, tw, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2), roughness=1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.1, 0.0, 0.025)),
        )
        log("added visual-only table extension under the arm bases")
    for name, eye, target, focal in OVERVIEWS:
        setattr(cfg.scene, f"ov_{name}", CameraCfg(
            prim_path=f"/World/OverviewCams/{name}",
            update_period=0.0,           # render every step
            height=args.ov_h, width=args.ov_w,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal, horizontal_aperture=20.955,
            ),
            offset=CameraCfg.OffsetCfg(pos=eye, rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        ))
    log("injected overview cameras:", ", ".join(n for n, *_ in OVERVIEWS))
    if args.joints_npz:
        npz = np.load(args.joints_npz, allow_pickle=True)
        capture_frame.jq = (npz["ik_left"].astype(np.float32), npz["ik_right"].astype(np.float32))
        log(f"arm posed from exact IK joints: {args.joints_npz} "
            f"(target={npz['target'] if 'target' in npz else '?'})")
    elif args.remap_arm:
        log(f"arm remap real->sim ARM_SIM_FROM_REAL={ARM_SIM_FROM_REAL}")

    env = FrankaOrcaBimanualEnv(cfg)
    env.reset(); env.reset()

    # ---- aim overview cameras (look-at) ----
    for name, eye, target, focal in OVERVIEWS:
        cam = env.scene[f"ov_{name}"]
        eyes = torch.tensor([eye], dtype=torch.float32, device=env.device)
        tgts = torch.tensor([target], dtype=torch.float32, device=env.device)
        cam.set_world_poses_from_view(eyes, tgts)
    log("aimed overview cameras via look-at")

    rows = []
    with h5py.File(args.h5, "r") as f:
        for k in ["observations/qpos_arm_left", "observations/qpos_arm_right",
                  "observations/qpos_hand_left", "observations/qpos_hand_right", ARIA_DS, OAKD_DS]:
            if k not in f:
                log("ERROR: h5 missing", k, "| has:", list(f.keys())); return
        for F in FRAMES:
            rows.append(capture_frame(env, f, F, args.out_dir))

    if len(rows) > 1:
        sheet = np.concatenate(rows, axis=0)
        cv2.imwrite(os.path.join(args.out_dir, "contact_sheet.png"),
                    cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        log("wrote", os.path.join(args.out_dir, "contact_sheet.png"))

    log("=" * 72); log("SNAPSHOT COMPLETE"); log("=" * 72)
    env.close(); simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("\n*** EXCEPTION ***"); log(traceback.format_exc()); os._exit(1)
