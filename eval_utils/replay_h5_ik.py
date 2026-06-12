"""Open-loop replay of a recorded episode by IK: dataset EE poses -> joint targets.

THE DATASET FORMAT (established 2026-06-11, Opus-panel verified): the 7-dim
`observations/qpos_arm_*` / `actions_arm_*` fields are END-EFFECTOR POSES
[x, y, z, qw, qx, qy, qz] in each arm's OWN BASE frame (meters, scalar-first unit
quaternion; ||q||=1 to machine precision on every frame) — NOT joint angles. The
old per-joint sign/offset remap (real_arm_to_sim) is therefore meaningless and is
NOT used here.

Pipeline per frame (the user-requested architecture — IK first, then JOINT commands,
never direct Cartesian control of the articulation):
  1. target pose (base frame) = h5 row  (base axes are parallel to world; sim
     articulation roots sit at the arm bases with identity rotation, so the base
     frame IS the controller frame — no extra transform, and NO calib y-flip).
  2. DifferentialIKController (dls) -> desired panda_joint1..7
  3. env.step() with the 48-dim action [qL(7), qR(7), handL(17), handR(17)] —
     absolute joint-position targets through the standard action terms.
  4. log achieved-vs-target EE error (pos mm / ori deg), write real|sim videos.

Reference point: --ee-body selects what the pose is assumed to refer to:
  flange (panda_link8, default) | root ({side}_root) | palm ({side}_palm).
If the replay shows a CONSTANT cartesian offset vs the real video, re-run with a
different --ee-body to pin the true reference point.

Run inside the Isaac container:
  /opt/IsaacLab/isaaclab.sh -p eval_utils/replay_h5_ik.py \
      --headless --enable_cameras --device cpu --h5 /app/20250826_111157.h5
"""

import argparse

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--h5", type=str, default="/app/20250826_111157.h5")
parser.add_argument("--out-dir", type=str, default="/output/ik_replay")
parser.add_argument("--max-steps", type=int, default=0, help="0 = full episode")
parser.add_argument("--stride", type=int, default=1, help="subsample recorded frames")
parser.add_argument("--source", choices=["qpos", "actions"], default="qpos",
                    help="qpos = measured EE pose (matches the videos); actions = teleop targets")
parser.add_argument("--warmup", type=int, default=150,
                    help="IK-convergence + camera-warmup steps toward the frame-0 pose")
parser.add_argument("--ee-body", choices=["flange", "root", "palm"], default="flange",
                    help="which body the dataset pose is assumed to describe")
parser.add_argument("--ee-offset-left", type=str, default=None,
                    help="constant offset 'px py pz qw qx qy qz': dataset frame expressed in the "
                         "ee body frame (T_body_to_data). IK then targets T_data * inv(T_off).")
parser.add_argument("--ee-offset-right", type=str, default=None)
# Orientation-delta models (dataset quat = wrist rotation RELATIVE to the episode-start /
# calibration pose; both arms start at ~identity which absolute body orientations cannot do):
#   --quat-pre  A: target quat = A * q_data        (delta applied in A's local frame)
#   --quat-post A: target quat = q_data * A        (delta applied in the base frame)
# Positions always pass through absolute. Estimators for both A's are reported at the end.
parser.add_argument("--quat-pre-left", type=str, default=None, help="'qw qx qy qz'")
parser.add_argument("--quat-pre-right", type=str, default=None)
parser.add_argument("--quat-post-left", type=str, default=None)
parser.add_argument("--quat-post-right", type=str, default=None)
parser.add_argument("--persp", dest="persp", action="store_true", default=True)
parser.add_argument("--no-persp", dest="persp", action="store_false")
parser.add_argument("--no-video", dest="video", action="store_false", default=True,
                    help="skip all video/image output (fast diagnostic runs)")
parser.add_argument("--settle-steps", type=int, default=1,
                    help="physics steps (with IK recompute) per recorded frame; >1 for estimation runs")
parser.add_argument("--joints-npz", type=str, default=None,
                    help="precomputed IK joint trajectories (eval_utils/export_ik_joints.py): skip "
                         "online IK and command these joints directly (exact offline solutions)")
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
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import (
    combine_frame_transforms, quat_apply as _quat_apply, quat_error_magnitude, quat_inv,
    quat_mul, subtract_frame_transforms,
)

import sim_envs  # noqa: F401  (registers the env)
from sim_envs.franka_orca_bimanual_cfg import (
    FrankaOrcaBimanualEnvCfg,
    LEFT_ARM_ORDER, RIGHT_ARM_ORDER, LEFT_HAND_ORDER, RIGHT_HAND_ORDER,
)
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv

os.makedirs(args.out_dir, exist_ok=True)
LOGP = os.path.join(args.out_dir, "ik_replay_log.txt")
_fh = open(LOGP, "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _fh.write(s + "\n"); _fh.flush()

ARIA_DS = "observations/images/aria_rgb_cam/color"
OAKD_DS = "observations/images/oakd_front_view/color"

PERSP_EYE = (1.15, -1.15, 0.95)
PERSP_TGT = (0.28, 0.0, 0.12)
PERSP_FOCAL = 14.0
PERSP_W, PERSP_H = 1920, 1440

EE_BODY = {
    "flange": lambda side: "panda_link8",
    "root": lambda side: f"{side}_root",
    "palm": lambda side: f"{side}_palm",
}[args.ee_body]


def label(img, text, color=(255, 255, 255)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def side_by_side(real_rgb, sim_rgb, cam_name, t, tsec):
    r = label(real_rgb, f"REAL  {cam_name}  t={t}  {tsec:5.2f}s", (120, 255, 120))
    s = label(sim_rgb, f"SIM(IK)  {cam_name}", (120, 200, 255))
    return np.concatenate([r, s], axis=1)


def draw_err(img, eL_mm, eL_deg, eR_mm, eR_deg):
    """Small top-right overlay: per-arm EE tracking error (achieved vs dataset target)."""
    out = img.copy()
    H, W = out.shape[:2]
    fs = max(0.45, round(W / 2600.0, 3)); th = max(1, int(fs * 2))
    lines = ["EE ERR (ach vs tgt)",
             f"L {eL_mm:6.1f}mm {eL_deg:5.1f}deg",
             f"R {eR_mm:6.1f}mm {eR_deg:5.1f}deg"]
    (cw, ch), _ = cv2.getTextSize(lines[0], cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = int(10 * fs); lh = ch + int(10 * fs)
    x0 = W - cw - 2 * pad; y0 = pad
    ov = out.copy()
    cv2.rectangle(ov, (x0, y0), (W - pad, y0 + 2 * pad + 3 * lh), (0, 0, 0), -1)
    out = cv2.addWeighted(ov, 0.55, out, 0.45, 0)
    colors = [(255, 230, 120), (140, 255, 140), (140, 200, 255)]
    ty = y0 + pad + ch
    for ln, c in zip(lines, colors):
        cv2.putText(out, ln, (x0 + pad, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, c, th, cv2.LINE_AA)
        ty += lh
    return out


class ArmIK:
    """Differential-IK wrapper for one arm of a 24-DOF arm+hand articulation."""

    def __init__(self, env, scene_key, side, arm_order, ee_offset=None,
                 quat_pre=None, quat_post=None):
        self.art = env.scene[scene_key]
        self.device = env.device
        self.joint_ids, resolved = self.art.find_joints(list(arm_order), preserve_order=True)
        assert list(resolved) == list(arm_order), f"{scene_key}: {resolved} != {arm_order}"
        body_name = EE_BODY(side)
        ids, names = self.art.find_bodies(body_name)
        assert len(ids) == 1, f"{scene_key}: body {body_name} -> {names}"
        self.body_idx = ids[0]
        # fixed-base articulation: jacobian row index = body index - 1
        self.jacobi_idx = self.body_idx - 1
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1, device=self.device,
        )
        # constant body->data offset: IK targets T_data * inv(T_off) so that
        # body * T_off == T_data when converged
        self.inv_off = None
        if ee_offset is not None:
            v = torch.tensor([float(x) for x in ee_offset.split()], device=self.device).unsqueeze(0)
            assert v.shape[-1] == 7, "ee-offset needs 7 numbers: px py pz qw qx qy qz"
            inv_q = quat_inv(v[:, 3:7])
            inv_p = -_quat_apply(inv_q, v[:, 0:3])
            self.inv_off = (inv_p, inv_q)
            log(f"  {scene_key}: applying ee offset T_off = {ee_offset}")
        def _q(s):
            if s is None:
                return None
            v = torch.tensor([float(x) for x in s.split()], device=self.device).unsqueeze(0)
            assert v.shape[-1] == 4, "quat needs 4 numbers: qw qx qy qz"
            return v / torch.norm(v, dim=-1, keepdim=True)
        self.quat_pre = _q(quat_pre)
        self.quat_post = _q(quat_post)
        if self.quat_pre is not None:
            log(f"  {scene_key}: target quat = A_pre * q_data, A_pre={quat_pre}")
        if self.quat_post is not None:
            log(f"  {scene_key}: target quat = q_data * A_post, A_post={quat_post}")
        # running estimate of T_off = dataset pose expressed in the CURRENT body frame
        self.off_samples_p, self.off_samples_q = [], []
        # delta-model estimators: A_pre = q_ach * inv(q_data); A_post = inv(q_data) * q_ach
        self.pre_samples, self.post_samples = [], []
        log(f"  {scene_key}: ee body '{body_name}' idx={self.body_idx}, arm joint ids={self.joint_ids}")

    def ee_pose_b(self):
        """Current EE pose in the articulation BASE frame (== dataset frame)."""
        ee_w = self.art.data.body_state_w[:, self.body_idx, 0:7]
        root_w = self.art.data.root_state_w[:, 0:7]
        pos_b, quat_b = subtract_frame_transforms(
            root_w[:, 0:3], root_w[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7])
        return pos_b, quat_b

    def body_target(self, data_pose_b):
        """Map a dataset pose to the BODY pose IK should reach (offset / delta models)."""
        if self.inv_off is not None:
            p, q = combine_frame_transforms(
                data_pose_b[:, 0:3], data_pose_b[:, 3:7], self.inv_off[0], self.inv_off[1])
            return torch.cat([p, q], dim=-1)
        q = data_pose_b[:, 3:7]
        if self.quat_pre is not None:
            q = quat_mul(self.quat_pre, q)
        if self.quat_post is not None:
            q = quat_mul(q, self.quat_post)
        return torch.cat([data_pose_b[:, 0:3], q], dim=-1)

    def solve(self, data_pose_b):
        """One differential-IK step toward the dataset pose -> (1,7) arm joints."""
        target = self.body_target(data_pose_b)
        self.ik.set_command(target)
        ee_pos_b, ee_quat_b = self.ee_pose_b()
        jac = self.art.root_physx_view.get_jacobians()[:, self.jacobi_idx, :, self.joint_ids]
        q = self.art.data.joint_pos[:, self.joint_ids]
        return self.ik.compute(ee_pos_b, ee_quat_b, jac, q)

    def error(self, data_pose_b, collect_offset=False):
        """(pos_err_m, ori_err_rad) of the CURRENT body pose vs the (offset-corrected) target.

        With collect_offset, also store a sample of T_off = dataset pose expressed in the
        CURRENT body frame; if this is constant across pose-diverse frames, it IS the rigid
        transform between the assumed body and the dataset's true reference frame."""
        target = self.body_target(data_pose_b)
        ee_pos_b, ee_quat_b = self.ee_pose_b()
        pe = torch.norm(ee_pos_b - target[:, 0:3], dim=-1)
        oe = quat_error_magnitude(ee_quat_b, target[:, 3:7])
        if collect_offset:
            op, oq = subtract_frame_transforms(
                ee_pos_b, ee_quat_b, data_pose_b[:, 0:3], data_pose_b[:, 3:7])
            self.off_samples_p.append(op[0].cpu().numpy())
            self.off_samples_q.append(oq[0].cpu().numpy())
            qd = data_pose_b[:, 3:7]
            self.pre_samples.append(quat_mul(ee_quat_b, quat_inv(qd))[0].cpu().numpy())
            self.post_samples.append(quat_mul(quat_inv(qd), ee_quat_b)[0].cpu().numpy())
        return float(pe[0]), float(oe[0])

    @staticmethod
    def _quat_stats(samples):
        Q = np.stack(samples)
        Q = Q * np.sign(np.sum(Q * Q[0], axis=1, keepdims=True))
        m = Q.mean(0); m /= np.linalg.norm(m)
        spread = np.degrees(2 * np.arccos(np.clip(np.abs(Q @ m), -1, 1)))
        return m, float(np.median(spread)), float(np.percentile(spread, 95))

    def offset_report(self, name):
        if not self.off_samples_p:
            return
        P = np.stack(self.off_samples_p); Q = np.stack(self.off_samples_q)
        Q = Q * np.sign(np.sum(Q * Q[0], axis=1, keepdims=True))  # hemisphere-align
        q_mean = Q.mean(0); q_mean /= np.linalg.norm(q_mean)
        ang_spread = np.degrees(2 * np.arccos(np.clip(np.abs(Q @ q_mean), -1, 1)))
        log(f"\n[{name}] estimated T_off (dataset frame in {args.ee_body} body frame), "
            f"{len(P)} samples:")
        log(f"  pos  mean {np.round(P.mean(0), 4)}  std {np.round(P.std(0), 4)}  (m)")
        log(f"  quat mean {np.round(q_mean, 4)}  angular spread p95 {np.percentile(ang_spread, 95):.1f} deg")
        log(f"  --ee-offset-{name} \"{P.mean(0)[0]:.5f} {P.mean(0)[1]:.5f} {P.mean(0)[2]:.5f} "
            f"{q_mean[0]:.5f} {q_mean[1]:.5f} {q_mean[2]:.5f} {q_mean[3]:.5f}\"")
        # delta-model estimates: whichever has the SMALLER spread is the right composition
        mp, med_p, p95_p = self._quat_stats(self.pre_samples)
        mq, med_q, p95_q = self._quat_stats(self.post_samples)
        log(f"  [delta models] A_pre  (q_t = A*q_data): mean {np.round(mp,4)}  spread med {med_p:.1f} / p95 {p95_p:.1f} deg")
        log(f"                 A_post (q_t = q_data*A): mean {np.round(mq,4)}  spread med {med_q:.1f} / p95 {p95_q:.1f} deg")
        log(f"  --quat-pre-{name} \"{mp[0]:.5f} {mp[1]:.5f} {mp[2]:.5f} {mp[3]:.5f}\"")
        log(f"  --quat-post-{name} \"{mq[0]:.5f} {mq[1]:.5f} {mq[2]:.5f} {mq[3]:.5f}\"")


def main():
    log("=" * 72)
    log(f"IK REPLAY  h5={args.h5}  source={args.source}  ee-body={args.ee_body}  stride={args.stride}")
    log("  dataset arm fields = EE poses [x,y,z,qw,qx,qy,qz] in the arm BASE frame")
    log("=" * 72)

    with h5py.File(args.h5, "r") as f:
        src = "observations/qpos_arm_{}" if args.source == "qpos" else "actions_arm_{}"
        pl = f[src.format("left")][:].astype(np.float32)
        pr = f[src.format("right")][:].astype(np.float32)
        hl = f["observations/qpos_hand_left"][:].astype(np.float32)
        hr = f["observations/qpos_hand_right"][:].astype(np.float32)
        T = min(len(pl), len(pr), len(hl), len(hr), f[ARIA_DS].shape[0], f[OAKD_DS].shape[0])
    nl = np.linalg.norm(pl[:, 3:7], axis=1); nr = np.linalg.norm(pr[:, 3:7], axis=1)
    log(f"episode T={T} @50Hz; quat-norm sanity: L[{nl.min():.6f},{nl.max():.6f}] R[{nr.min():.6f},{nr.max():.6f}]")
    pl[:, 3:7] /= nl[:, None]; pr[:, 3:7] /= nr[:, None]  # exact unit (paranoia)

    cfg = FrankaOrcaBimanualEnvCfg()
    cfg.sim.device = args.device
    if args.persp:
        cfg.scene.ov_persp = CameraCfg(
            prim_path="/World/OverviewCams/persp",
            update_period=0.0, height=PERSP_H, width=PERSP_W,
            spawn=sim_utils.PinholeCameraCfg(focal_length=PERSP_FOCAL, horizontal_aperture=20.955),
            offset=CameraCfg.OffsetCfg(pos=PERSP_EYE, rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        )
    env = FrankaOrcaBimanualEnv(cfg)
    env.reset(); env.reset()
    if args.persp:
        env.scene["ov_persp"].set_world_poses_from_view(
            torch.tensor([PERSP_EYE], dtype=torch.float32, device=env.device),
            torch.tensor([PERSP_TGT], dtype=torch.float32, device=env.device))

    log("[IK setup]")
    ikL = ArmIK(env, "left_arm_hand", "left", LEFT_ARM_ORDER, args.ee_offset_left,
                args.quat_pre_left, args.quat_post_left)
    ikR = ArmIK(env, "right_arm_hand", "right", RIGHT_ARM_ORDER, args.ee_offset_right,
                args.quat_pre_right, args.quat_post_right)
    # sanity: articulation roots must sit at the arm bases with identity rotation,
    # otherwise base frame != dataset frame and the whole replay is garbage
    for nm, ik in [("left", ikL), ("right", ikR)]:
        r = ik.art.data.root_state_w[0, 0:7].cpu().numpy()
        log(f"  {nm} root world pose = pos {np.round(r[:3],4)} quat {np.round(r[3:7],4)}")

    def step_with(qL, qR, h_l, h_r):
        a = torch.zeros((1, 48), dtype=torch.float32, device=env.device)
        a[0, 0:7] = qL; a[0, 7:14] = qR
        a[0, 14:31] = torch.tensor(h_l, device=env.device)
        a[0, 31:48] = torch.tensor(h_r, device=env.device)
        return env.step(a)

    jql = jqr = None
    if args.joints_npz:
        npz = np.load(args.joints_npz, allow_pickle=True)
        jql = npz["ik_left"].astype(np.float32); jqr = npz["ik_right"].astype(np.float32)
        assert len(jql) >= T and len(jqr) >= T, f"npz has {len(jql)} frames < episode {T}"
        log(f"using precomputed IK joints from {args.joints_npz} "
            f"(offline residuals: pos max {npz['pos_err_left'].max()*1000:.2f}/"
            f"{npz['pos_err_right'].max()*1000:.2f} mm)")

    tgtL = torch.tensor(pl[0], device=env.device).unsqueeze(0)
    tgtR = torch.tensor(pr[0], device=env.device).unsqueeze(0)

    def joints_for(t):
        if jql is not None:
            return (torch.tensor(jql[t], device=env.device),
                    torch.tensor(jqr[t], device=env.device))
        return ikL.solve(tgtL)[0], ikR.solve(tgtR)[0]

    # ---- warmup: drive toward the frame-0 solution (also hydrates cameras) ----
    if jql is not None:  # teleport straight to the exact frame-0 IK solution
        for skey, ik, q0 in [("left_arm_hand", ikL, jql[0]), ("right_arm_hand", ikR, jqr[0])]:
            art = env.scene[skey]
            pos = torch.tensor(q0, device=env.device).unsqueeze(0)
            art.write_joint_state_to_sim(pos, torch.zeros_like(pos), joint_ids=ik.joint_ids)
            art.set_joint_position_target(pos, joint_ids=ik.joint_ids)
        env.scene.write_data_to_sim()
        log("teleported both arms to the frame-0 IK solution")
    obs = None
    for k in range(max(1, args.warmup)):
        qL, qR = joints_for(0)
        obs, _ = step_with(qL, qR, hl[0], hr[0])
        if k in (0, args.warmup // 2, args.warmup - 1):
            eL = ikL.error(tgtL); eR = ikR.error(tgtR)
            log(f"  warmup {k:4d}: L {eL[0]*1000:7.1f}mm {np.degrees(eL[1]):6.1f}deg | "
                f"R {eR[0]*1000:7.1f}mm {np.degrees(eR[1]):6.1f}deg")

    # ---- video writers ----
    aria_w = oakd_w = comb_w = persp_w = None
    COMB_W = 1280
    aria_row_h = int(round(480 * COMB_W / (640 * 2)))
    oakd_row_h = int(round(540 * COMB_W / (960 * 2)))
    if args.video:
        fps = 50.0 / max(1, args.stride)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        aria_w = cv2.VideoWriter(os.path.join(args.out_dir, "ik_aria_real_vs_sim.mp4"), fourcc, fps, (640 * 2, 480))
        oakd_w = cv2.VideoWriter(os.path.join(args.out_dir, "ik_oakd_real_vs_sim.mp4"), fourcc, fps, (960 * 2, 540))
        comb_w = cv2.VideoWriter(os.path.join(args.out_dir, "ik_replay_sidebyside.mp4"), fourcc, fps,
                                 (COMB_W, aria_row_h + oakd_row_h))
        persp_w = cv2.VideoWriter(os.path.join(args.out_dir, "ik_replay_persp.mp4"), fourcc, fps,
                                  (PERSP_W, PERSP_H)) if args.persp else None
        for w, nm in [(aria_w, "aria"), (oakd_w, "oakd"), (comb_w, "comb")] + ([(persp_w, "persp")] if persp_w else []):
            if not w.isOpened():
                log(f"ERROR: VideoWriter {nm} failed"); return

    idxs = list(range(0, T, args.stride))
    if args.max_steps > 0:
        idxs = idxs[: args.max_steps]
    snap_at = {idxs[int(len(idxs) * p)] for p in (0.0, 0.25, 0.5, 0.75)} | {idxs[-1]}

    errs = np.zeros((len(idxs), 4), dtype=np.float32)  # Lmm, Ldeg, Rmm, Rdeg
    BATCH = 128
    real_aria = real_oakd = None; batch_lo = -10**9
    f = h5py.File(args.h5, "r")
    try:
        for n, t in enumerate(idxs):
            if not (batch_lo <= t < batch_lo + (0 if real_aria is None else real_aria.shape[0])):
                hi = min(t + BATCH, T)
                real_aria = f[ARIA_DS][t:hi]; real_oakd = f[OAKD_DS][t:hi]; batch_lo = t
            rj = t - batch_lo

            tgtL = torch.tensor(pl[t], device=env.device).unsqueeze(0)
            tgtR = torch.tensor(pr[t], device=env.device).unsqueeze(0)
            for _ in range(max(1, args.settle_steps)):
                qL, qR = joints_for(t)
                if jql is not None:  # exact placement: teleport arm joints (no tracking lag)
                    for skey, ik, qv in [("left_arm_hand", ikL, qL), ("right_arm_hand", ikR, qR)]:
                        art = env.scene[skey]
                        pos = qv.unsqueeze(0)
                        art.write_joint_state_to_sim(pos, torch.zeros_like(pos), joint_ids=ik.joint_ids)
                    env.scene.write_data_to_sim()
                obs, _ = step_with(qL, qR, hl[t], hr[t])

            eL = ikL.error(tgtL, collect_offset=True); eR = ikR.error(tgtR, collect_offset=True)
            errs[n] = (eL[0] * 1000, np.degrees(eL[1]), eR[0] * 1000, np.degrees(eR[1]))

            if args.video:
                sim_aria = obs["policy"]["aria_rgb_cam"][0, ..., :3].cpu().numpy().astype(np.uint8)
                sim_oakd = obs["policy"]["oakd_front_view"][0, ..., :3].cpu().numpy().astype(np.uint8)
                tsec = t / 50.0
                a_sbs = draw_err(side_by_side(real_aria[rj], sim_aria, "aria", t, tsec), *errs[n])
                o_sbs = draw_err(side_by_side(real_oakd[rj], sim_oakd, "oakd", t, tsec), *errs[n])
                aria_w.write(cv2.cvtColor(a_sbs, cv2.COLOR_RGB2BGR))
                oakd_w.write(cv2.cvtColor(o_sbs, cv2.COLOR_RGB2BGR))
                comb = np.concatenate([cv2.resize(a_sbs, (COMB_W, aria_row_h)),
                                       cv2.resize(o_sbs, (COMB_W, oakd_row_h))], axis=0)
                comb_w.write(cv2.cvtColor(comb, cv2.COLOR_RGB2BGR))
                if persp_w is not None:
                    pim = env.scene["ov_persp"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
                    persp_w.write(cv2.cvtColor(draw_err(pim, *errs[n]), cv2.COLOR_RGB2BGR))
                if t in snap_at:
                    cv2.imwrite(os.path.join(args.out_dir, f"ik_sbs_t{t}.png"), cv2.cvtColor(comb, cv2.COLOR_RGB2BGR))
            if n % 400 == 0:
                log(f"  t={t:5d} ({n}/{len(idxs)})  L {errs[n,0]:6.1f}mm {errs[n,1]:5.1f}deg | "
                    f"R {errs[n,2]:6.1f}mm {errs[n,3]:5.1f}deg")
    finally:
        f.close()
        for w in (aria_w, oakd_w, comb_w, persp_w):
            if w is not None:
                w.release()

    log("\n==== EE TRACKING ERROR (achieved vs target, after stepping) ====")
    for i, nm in [(0, "L pos mm"), (1, "L ori deg"), (2, "R pos mm"), (3, "R ori deg")]:
        v = errs[:, i]
        log(f"  {nm}: mean {v.mean():7.2f}  median {np.median(v):7.2f}  p95 {np.percentile(v,95):7.2f}  max {v.max():7.2f}")
    np.save(os.path.join(args.out_dir, "ik_errors.npy"), errs)
    ikL.offset_report("left")
    ikR.offset_report("right")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ts = np.array(idxs) / 50.0
        axes[0].plot(ts, errs[:, 0], label="left"); axes[0].plot(ts, errs[:, 2], label="right")
        axes[0].set_ylabel("EE pos err (mm)"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(ts, errs[:, 1], label="left"); axes[1].plot(ts, errs[:, 3], label="right")
        axes[1].set_ylabel("EE ori err (deg)"); axes[1].set_xlabel("episode time (s)"); axes[1].grid(alpha=0.3)
        fig.suptitle(f"IK replay tracking error  (source={args.source}, ee-body={args.ee_body})")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "ik_error_curves.png"), dpi=110)
        log("wrote ik_error_curves.png")
    except Exception as e:  # matplotlib is non-critical
        log(f"(skip error plot: {e})")

    log("=" * 72); log("IK REPLAY COMPLETE ->", args.out_dir); log("=" * 72)
    env.close(); simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("\n*** EXCEPTION ***"); log(traceback.format_exc()); os._exit(1)
