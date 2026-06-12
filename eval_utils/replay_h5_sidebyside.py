"""Open-loop h5 replay with a time-aligned REAL-vs-SIM side-by-side video.

Validates the actuator mapping end-to-end: feeds the *recorded* h5 actions (teleop
joint targets) into the FrankaOrcaBimanual sim as ABSOLUTE joint-position targets
(the training action layout [left_arm(7), right_arm(7), left_hand(17), right_hand(17)]),
having first teleported the sim to the recorded first-frame state (qpos[0]). While it
replays it renders both sim cameras (aria 640x480, oak-d 960x540) AND reads the matching
*real* camera frames straight out of the same h5, and writes them concatenated
side-by-side, frame-for-frame, so you can eyeball whether the sim reproduces the demo.

This is the same mechanical test as eval_utils/replay_h5.py, but it (a) sets the initial
condition from the first h5 frame and (b) produces the real-vs-sim comparison in one pass.

Every output frame carries a top-right table of the 14 arm joints (j1..j7 x both
robots) showing, in radians, the value ACTUALLY WRITTEN that frame (cmd, post j4/j6
remap) next to the value READ BACK from the sim (sim) — so you can see exactly what
is commanded vs achieved per joint (hand joints excluded). See draw_joint_table().

Outputs (to --out-dir, default /output):
  aria_real_vs_sim.mp4   [real aria | sim aria]   (1280x480)
  oakd_real_vs_sim.mp4   [real oak-d | sim oak-d] (1920x540)
  replay_sidebyside.mp4  2x2 grid (aria row over oak-d row), labelled
  replay_persp.mp4       high-res perspective overview (1920x1440)
  replay_sbs_log.txt     joint-mapping check + commanded-vs-achieved tracking

Arm convention: the recorded arm joints use a different Franka convention than the sim
URDF (j4 ~ +0.9 is outside the sim limit), so real_arm_to_sim() applies the configured
ARM_SIM_FROM_REAL offsets by default. Disable with --no-remap-arm to SEE the broken
(clamped) case.

Run inside the Isaac container, e.g. (docker, this machine):
  /workspace/isaaclab/isaaclab.sh -p eval_utils/replay_h5_sidebyside.py \
      --headless --enable_cameras --device cuda:0 --h5 /app/20250826_111157.h5
"""

import argparse

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--h5", type=str, default="/app/20250826_111157.h5")
parser.add_argument("--out-dir", type=str, default="/output")
parser.add_argument("--max-steps", type=int, default=0, help="0 = full episode")
parser.add_argument("--stride", type=int, default=1, help="subsample recorded frames")
parser.add_argument("--warmup", type=int, default=40, help="settle/camera-warmup steps at frame 0")
parser.add_argument("--init-to-first", dest="init_to_first", action="store_true", default=True,
                    help="teleport sim joints to the recorded first-frame state (qpos[0]) before replay")
parser.add_argument("--no-init-to-first", dest="init_to_first", action="store_false")
parser.add_argument("--remap-arm", dest="remap_arm", action="store_true", default=True,
                    help="apply real->sim arm joint-convention remap")
parser.add_argument("--no-remap-arm", dest="remap_arm", action="store_false")
parser.add_argument("--persp", dest="persp", action="store_true", default=True,
                    help="also render a high-res perspective overview video (replay_persp.mp4)")
parser.add_argument("--no-persp", dest="persp", action="store_false")
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
    get_left_arm_joint_pos, get_right_arm_joint_pos,
    get_left_hand_joint_pos, get_right_hand_joint_pos,
)
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv

os.makedirs(args.out_dir, exist_ok=True)
LOGP = os.path.join(args.out_dir, "replay_sbs_log.txt")
_fh = open(LOGP, "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _fh.write(s + "\n"); _fh.flush()

# real camera datasets in the h5 and their sim obs key + native (W,H)
ARIA_DS = "observations/images/aria_rgb_cam/color"
OAKD_DS = "observations/images/oakd_front_view/color"

# High-res perspective overview camera (same pose as the "persp" view in
# eval_utils/snapshot_frame0_multiview.py). Free camera looking at the whole rig.
PERSP_EYE = (1.15, -1.15, 0.95)
PERSP_TGT = (0.28, 0.0, 0.12)
PERSP_FOCAL = 14.0
PERSP_W, PERSP_H = 1920, 1440


def label(img, text, org=(6, 22), color=(255, 255, 255)):
    """Draw a small dark banner with `text` at the top-left of an RGB frame (in place-ish)."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


_JOINT_LABELS = [f"j{i}" for i in range(1, 8)]


def draw_joint_table(img, lc, ls, rc, rs):
    """Overlay a top-right table of the 7 arm joints for BOTH robots (no hand joints).

    Per joint shows, in radians:
      Lcmd/Rcmd = the value ACTUALLY WRITTEN to the sim this frame (post j4/j6 remap)
      Lsim/Rsim = the value READ BACK from the articulation (achieved; reveals clamping)

    lc, ls, rc, rs are length-7 arrays (left cmd/sim, right cmd/sim). Columns are
    placed at fixed x-offsets (cv2 fonts aren't monospace) so they align. Font scales
    with frame width so it stays legible on the small camera videos and the big persp.
    """
    out = img.copy()
    H, W = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.42, round(W / 2400.0, 3))
    th = max(1, int(round(fs * 2)))
    (cw, ch), _ = cv2.getTextSize("+0.000", font, fs, th)
    (lw, _), _ = cv2.getTextSize("j7", font, fs, th)
    pad = int(round(10 * fs))
    colgap = int(round(cw * 0.45))
    label_w = lw + colgap
    cell_w = cw + colgap
    headers = ["Lcmd", "Lsim", "Rcmd", "Rsim"]
    line_h = ch + int(round(11 * fs))
    n_lines = 2 + 7  # title + header + 7 joints
    table_w = label_w + len(headers) * cell_w + pad * 2
    table_h = pad * 2 + n_lines * line_h
    x0 = W - table_w - pad
    y0 = pad

    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + table_w, y0 + table_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    tx = x0 + pad
    ty = y0 + pad + ch
    GREEN, BLUE = (140, 255, 140), (140, 200, 255)  # left, right (RGB)
    cv2.putText(out, "ARM JOINTS (rad)", (tx, ty), font, fs, (255, 230, 120), th, cv2.LINE_AA)
    ty += line_h
    cx = tx + label_w
    for i, h in enumerate(headers):
        cv2.putText(out, h, (cx, ty), font, fs, GREEN if i < 2 else BLUE, th, cv2.LINE_AA)
        cx += cell_w
    ty += line_h
    for j in range(7):
        cv2.putText(out, _JOINT_LABELS[j], (tx, ty), font, fs, (220, 220, 220), th, cv2.LINE_AA)
        cx = tx + label_w
        for i, v in enumerate((lc[j], ls[j], rc[j], rs[j])):
            cv2.putText(out, f"{v:+.3f}", (cx, ty), font, fs, GREEN if i < 2 else BLUE, th, cv2.LINE_AA)
            cx += cell_w
        ty += line_h
    return out


def side_by_side(real_rgb, sim_rgb, cam_name, t, tsec):
    """[REAL | SIM] for one camera; both are HxWx3 RGB uint8 at the same native res."""
    r = label(real_rgb, f"REAL  {cam_name}  t={t}  {tsec:5.2f}s", color=(120, 255, 120))
    s = label(sim_rgb, f"SIM   {cam_name}", color=(120, 200, 255))
    return np.concatenate([r, s], axis=1)


def main():
    log("=" * 72)
    log("REAL-vs-SIM open-loop replay:", args.h5)
    log(f"  init_to_first={args.init_to_first}  remap_arm={args.remap_arm}  stride={args.stride}")
    log("=" * 72)

    with h5py.File(args.h5, "r") as f:
        need = ["actions_arm_left", "actions_arm_right", "actions_hand_left", "actions_hand_right",
                ARIA_DS, OAKD_DS]
        miss = [k for k in need if k not in f]
        if miss:
            log("ERROR: h5 missing keys:", miss, "| has:", list(f.keys())); return
        al = f["actions_arm_left"][:]; ar = f["actions_arm_right"][:]
        hl = f["actions_hand_left"][:]; hr = f["actions_hand_right"][:]
        # recorded first-frame STATE (initial condition)
        q_al = f["observations/qpos_arm_left"][0]; q_ar = f["observations/qpos_arm_right"][0]
        q_hl = f["observations/qpos_hand_left"][0]; q_hr = f["observations/qpos_hand_right"][0]
        T = min(len(al), len(ar), len(hl), len(hr), f[ARIA_DS].shape[0], f[OAKD_DS].shape[0])
        log(f"episode T={T}; arm{al.shape[1]} hand{hl.shape[1]} per side; "
            f"aria{f[ARIA_DS].shape[1:]} oakd{f[OAKD_DS].shape[1:]}")
    al = al[:T].astype(np.float32); ar = ar[:T].astype(np.float32)
    hl = hl[:T].astype(np.float32); hr = hr[:T].astype(np.float32)

    if args.remap_arm:
        log(f"arm remap real->sim (per-arm sign+offset) ARM_SIM_FROM_REAL={ARM_SIM_FROM_REAL}")
        al = np.stack([real_arm_to_sim(a, "left") for a in al]).astype(np.float32)
        ar = np.stack([real_arm_to_sim(a, "right") for a in ar]).astype(np.float32)
        q_al = real_arm_to_sim(q_al, "left").astype(np.float32)
        q_ar = real_arm_to_sim(q_ar, "right").astype(np.float32)
    # full 48-dim absolute action in training layout [L_arm, R_arm, L_hand, R_hand]
    actions = np.concatenate([al, ar, hl, hr], axis=-1).astype(np.float32)

    cfg = FrankaOrcaBimanualEnvCfg()
    cfg.sim.device = args.device
    if args.persp:
        cfg.scene.ov_persp = CameraCfg(
            prim_path="/World/OverviewCams/persp",
            update_period=0.0,           # render every step
            height=PERSP_H, width=PERSP_W,
            spawn=sim_utils.PinholeCameraCfg(focal_length=PERSP_FOCAL, horizontal_aperture=20.955),
            offset=CameraCfg.OffsetCfg(pos=PERSP_EYE, rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        )
        log(f"injected high-res persp overview camera {PERSP_W}x{PERSP_H}")
    env = FrankaOrcaBimanualEnv(cfg)
    env.reset(); env.reset()

    # ---- actuator-mapping check: articulation joint order vs training order ----
    log("\n[joint-mapping check]")
    for skey, side in [("left_arm_hand", "left"), ("right_arm_hand", "right")]:
        art = env.scene[skey]
        actual = list(art.joint_names)
        want = (LEFT_ARM_ORDER + LEFT_HAND_ORDER) if side == "left" else (RIGHT_ARM_ORDER + RIGHT_HAND_ORDER)
        ids, resolved = art.find_joints(list(want), preserve_order=True)
        ok = list(resolved) == list(want)
        log(f"  {skey}: {len(actual)} dofs; find_joints(preserve_order) resolves training order: {ok}")
        if actual != want:
            log(f"    NOTE Isaac internal order != training order (expected; we address BY NAME). "
                f"raw[:7]={actual[:7]}")

    # ---- initial condition: teleport to recorded first-frame state ----
    if args.init_to_first:
        for skey, side, qarm, qhand in [
            ("left_arm_hand", "left", q_al, q_hl),
            ("right_arm_hand", "right", q_ar, q_hr),
        ]:
            art = env.scene[skey]
            names = (LEFT_ARM_ORDER + LEFT_HAND_ORDER) if side == "left" else (RIGHT_ARM_ORDER + RIGHT_HAND_ORDER)
            ids, _ = art.find_joints(list(names), preserve_order=True)
            vals = np.concatenate([qarm, qhand]).astype(np.float32)
            pos = torch.tensor(vals, device=env.device).unsqueeze(0)
            vel = torch.zeros_like(pos)
            # joint_ids in the same list form the cfg uses for find_joints indexing
            art.write_joint_state_to_sim(pos, vel, joint_ids=ids)
            art.set_joint_position_target(pos, joint_ids=ids)
        env.scene.write_data_to_sim()
        log("teleported both arms to recorded frame-0 qpos (arm remapped)")

    # aim the high-res perspective overview camera (look-at), if enabled
    if args.persp:
        env.scene["ov_persp"].set_world_poses_from_view(
            torch.tensor([PERSP_EYE], dtype=torch.float32, device=env.device),
            torch.tensor([PERSP_TGT], dtype=torch.float32, device=env.device),
        )
        log("aimed persp overview camera")

    # warmup: hold the first action so physics settles and camera buffers hydrate
    a0 = torch.tensor(actions[0], dtype=torch.float32, device=env.device).unsqueeze(0)
    for _ in range(max(1, args.warmup)):
        obs, _ = env.step(a0)

    # ---- video writers (stream; no full-episode accumulation) ----
    fps = 50.0 / max(1, args.stride)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    aria_w = cv2.VideoWriter(os.path.join(args.out_dir, "aria_real_vs_sim.mp4"), fourcc, fps, (640 * 2, 480))
    oakd_w = cv2.VideoWriter(os.path.join(args.out_dir, "oakd_real_vs_sim.mp4"), fourcc, fps, (960 * 2, 540))
    # combined 2x2: both rows resized to a common width COMB_W
    COMB_W = 1280
    aria_row_h = int(round(480 * COMB_W / (640 * 2)))   # 480
    oakd_row_h = int(round(540 * COMB_W / (960 * 2)))   # 360
    comb_h = aria_row_h + oakd_row_h
    comb_w = cv2.VideoWriter(os.path.join(args.out_dir, "replay_sidebyside.mp4"), fourcc, fps, (COMB_W, comb_h))
    persp_w = None
    if args.persp:
        persp_w = cv2.VideoWriter(os.path.join(args.out_dir, "replay_persp.mp4"), fourcc, fps, (PERSP_W, PERSP_H))
    writers = [(aria_w, "aria"), (oakd_w, "oakd"), (comb_w, "combined")]
    if persp_w is not None:
        writers.append((persp_w, "persp"))
    for w, nm in writers:
        if not w.isOpened():
            log(f"ERROR: VideoWriter for {nm} failed to open"); return

    idxs = list(range(0, T, args.stride))
    if args.max_steps and args.max_steps > 0:
        idxs = idxs[: args.max_steps]
    track_at = {int(len(idxs) * p) for p in (0.0, 0.25, 0.5, 0.75, 0.99)}

    # prefetch real frames in batches (HDF5 list-indexing is much faster than per-frame seeks)
    BATCH = 128
    real_aria = real_oakd = None
    batch_lo = -1
    fcount = 0
    f = h5py.File(args.h5, "r")
    try:
        for n, t in enumerate(idxs):
            if not (batch_lo <= t < batch_lo + (real_aria.shape[0] if real_aria is not None else 0)):
                hi = min(t + BATCH, T)
                real_aria = f[ARIA_DS][t:hi]     # (b,480,640,3) RGB uint8
                real_oakd = f[OAKD_DS][t:hi]     # (b,540,960,3)
                batch_lo = t
            rj = t - batch_lo

            act = torch.tensor(actions[t], dtype=torch.float32, device=env.device).unsqueeze(0)
            obs, _ = env.step(act)
            sim_aria = obs["policy"]["aria_rgb_cam"][0, ..., :3].cpu().numpy().astype(np.uint8)
            sim_oakd = obs["policy"]["oakd_front_view"][0, ..., :3].cpu().numpy().astype(np.uint8)

            tsec = t / 50.0
            aria_sbs = side_by_side(real_aria[rj], sim_aria, "aria", t, tsec)   # 480x1280
            oakd_sbs = side_by_side(real_oakd[rj], sim_oakd, "oakd", t, tsec)   # 540x1920

            # arm-joint table values: cmd = ACTUALLY WRITTEN this frame (post j4/j6 remap),
            # sim = read back from the articulation (achieved). Hand joints excluded.
            la_cmd = actions[t, 0:7]; ra_cmd = actions[t, 7:14]
            la_sim = get_left_arm_joint_pos(env)[0].cpu().numpy()
            ra_sim = get_right_arm_joint_pos(env)[0].cpu().numpy()

            aria_w.write(cv2.cvtColor(draw_joint_table(aria_sbs, la_cmd, la_sim, ra_cmd, ra_sim), cv2.COLOR_RGB2BGR))
            oakd_w.write(cv2.cvtColor(draw_joint_table(oakd_sbs, la_cmd, la_sim, ra_cmd, ra_sim), cv2.COLOR_RGB2BGR))
            # build the 2x2 from the PLAIN sbs frames, then draw ONE table on the combined
            # image (so it isn't doubled across rows or squished by the resize).
            comb = np.concatenate([
                cv2.resize(aria_sbs, (COMB_W, aria_row_h)),
                cv2.resize(oakd_sbs, (COMB_W, oakd_row_h)),
            ], axis=0)
            comb = draw_joint_table(comb, la_cmd, la_sim, ra_cmd, ra_sim)
            comb_w.write(cv2.cvtColor(comb, cv2.COLOR_RGB2BGR))
            if persp_w is not None:
                persp_rgb = env.scene["ov_persp"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
                persp_rgb = label(persp_rgb, f"SIM persp  t={t}  {tsec:5.2f}s", color=(255, 230, 120))
                persp_rgb = draw_joint_table(persp_rgb, la_cmd, la_sim, ra_cmd, ra_sim)
                persp_w.write(cv2.cvtColor(persp_rgb, cv2.COLOR_RGB2BGR))
            fcount += 1

            if n % 250 == 0:
                log(f"  ... {n}/{len(idxs)} frames")
            if n in track_at:
                arm_cmd = actions[t, 0:7]
                arm_ach = get_left_arm_joint_pos(env)[0].cpu().numpy()
                hand_cmd = actions[t, 14:31]
                hand_ach = get_left_hand_joint_pos(env)[0].cpu().numpy()
                log(f"[t={t:>4} {tsec:5.1f}s] L_arm max|err|={np.abs(arm_ach-arm_cmd).max():.3f} "
                    f"L_hand max|err|={np.abs(hand_ach-hand_cmd).max():.3f}")
    finally:
        f.close()

    aria_w.release(); oakd_w.release(); comb_w.release()
    if persp_w is not None:
        persp_w.release()
    log(f"\nwrote {fcount} frames @ {fps:.1f} fps to {args.out_dir}:")
    log("  aria_real_vs_sim.mp4  [real|sim] 1280x480")
    log("  oakd_real_vs_sim.mp4  [real|sim] 1920x540")
    log(f"  replay_sidebyside.mp4 2x2 {COMB_W}x{comb_h}")
    if persp_w is not None:
        log(f"  replay_persp.mp4      perspective overview {PERSP_W}x{PERSP_H}")
    log("=" * 72); log("REPLAY COMPLETE"); log("=" * 72)
    env.close(); simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("\n*** EXCEPTION ***"); log(traceback.format_exc()); _fh.flush(); os._exit(1)
