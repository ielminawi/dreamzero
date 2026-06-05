#!/usr/bin/env python3
"""
World-model video-prediction validation against ground-truth LeRobot episodes.

For each (episode, start_frame) "clip", this:
  1. Reads the real ground-truth camera frames from the LeRobot dataset.
  2. Runs DreamZero's proven action-free, video-conditioned autoregressive
     prediction (the exact contract used by scripts/test_generate_video.py and
     the websocket eval server -> GrootSimPolicy.lazy_joint_forward_causal).
  3. Decodes the predicted latent video to pixels.
  4. Temporally aligns predicted frames to the ground-truth frames (small offset
     search maximising SSIM) and computes per-frame PSNR / SSIM / MSE per camera
     view.
  5. Writes a side-by-side [GT | PRED] mp4 per camera and a per-clip metrics json.

The model is loaded ONCE and reused across all clips (fast, token/GPU cheap).

This deliberately mirrors test_generate_video.py for the model-call contract so
that the prediction path is identical to what produced the existing eval videos.
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

# ── per-embodiment constants (mirrors scripts/test_generate_video.py) ───────────
CAMERA_KEYS = ["video.aria_rgb_cam", "video.oakd_front_view"]
# ConcatTransform.video_concat_order for franka_orca_bimanual == [aria, oakd];
# the decoded composite is the two views side-by-side along WIDTH in that order.
CONCAT_ORDER = ["aria_rgb_cam", "oakd_front_view"]
DATASET_CAM = {  # video.<x> -> dataset video key
    "video.aria_rgb_cam": "observation.images.aria_rgb_cam",
    "video.oakd_front_view": "observation.images.oakd_front_view",
}
STATE_KEYS_DIMS = {
    "state.left_arm_joint_pos": 7,
    "state.right_arm_joint_pos": 7,
    "state.left_hand_joint_pos": 17,
    "state.right_hand_joint_pos": 17,
}
LANG_KEY = "annotation.task"

log = logging.getLogger("validate_wm")


# ── GT loading ──────────────────────────────────────────────────────────────
def read_gt_window(dataset_dir, ep_idx, cam_keys, start, n_needed):
    """Return {cam_key: (n_needed,H,W,3) uint8} read from the dataset mp4s."""
    import decord

    out = {}
    for cam_key in cam_keys:
        dk = DATASET_CAM[cam_key]
        path = os.path.join(
            dataset_dir, "videos", "chunk-000", dk, f"episode_{ep_idx:06d}.mp4"
        )
        vr = decord.VideoReader(path)
        total = len(vr)
        end = min(start + n_needed, total)
        idxs = list(range(start, end))
        frames = vr.get_batch(idxs).asnumpy()  # (n,H,W,3) uint8
        if len(frames) < n_needed:  # pad by repeating last frame
            pad = np.repeat(frames[-1:], n_needed - len(frames), axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        out[cam_key] = frames
    return out


def build_obs(video_buffers, step, frames_per_chunk, instruction):
    """One inference-step observation dict (identical logic to test_generate_video)."""
    if step == 0:
        n_frames, start_idx = 1, 0
    else:
        n_frames = frames_per_chunk
        start_idx = 1 + (step - 1) * frames_per_chunk

    obs = {}
    for cam_key, all_frames in video_buffers.items():
        total = len(all_frames)
        if start_idx >= total:
            raise ValueError(f"buffer too short for step {step}: idx {start_idx} >= {total}")
        end_idx = min(start_idx + n_frames, total)
        selected = all_frames[start_idx:end_idx]
        if len(selected) < n_frames:
            pad = np.repeat(selected[-1:], n_frames - len(selected), axis=0)
            selected = np.concatenate([selected, pad], axis=0)
        obs[cam_key] = selected
    for key, dim in STATE_KEYS_DIMS.items():
        obs[key] = np.zeros((1, dim), dtype=np.float64)
    obs[LANG_KEY] = instruction
    return obs


def decode_latents(policy, latent_list):
    """Decode a list of latent video tensors -> (T,H,W,C) uint8 (mirrors test script)."""
    from einops import rearrange

    ah = policy.trained_model.action_head
    video_cat = torch.cat(latent_list, dim=2)
    with torch.no_grad():
        frames = ah.vae.decode(
            video_cat,
            tiled=ah.tiled,
            tile_size=(ah.tile_size_height, ah.tile_size_width),
            tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
        )
    frames = rearrange(frames, "B C T H W -> B T H W C")[0]
    frames = ((frames.float() + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
    return frames


# ── metrics ───────────────────────────────────────────────────────────────────
def resize_to(img, h, w):
    import cv2

    if img.shape[0] == h and img.shape[1] == w:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def frame_metrics(gt, pred):
    """PSNR, SSIM, MSE between two uint8 HxWx3 frames (same size)."""
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    g = gt.astype(np.float64)
    p = pred.astype(np.float64)
    mse = float(np.mean((g - p) ** 2))
    ps = float(psnr(gt, pred, data_range=255))
    ss = float(ssim(gt, pred, data_range=255, channel_axis=2))
    return dict(mse=mse, psnr=ps, ssim=ss)


def best_offset(gt_views, pred_views, max_off):
    """Pick temporal offset o in [0,max_off] maximising mean SSIM of view 0.

    gt_views/pred_views: lists over cameras of (T,H,W,3) arrays already same-size.
    Returns (offset, n_overlap).
    """
    from skimage.metrics import structural_similarity as ssim

    P = pred_views[0].shape[0]
    G = gt_views[0].shape[0]
    best_o, best_s = 0, -1e9
    for o in range(0, max_off + 1):
        n = min(P, G - o)
        if n <= 0:
            continue
        ss = []
        for k in range(0, n, max(1, n // 8)):  # sample up to ~8 frames for speed
            ss.append(ssim(gt_views[0][o + k], pred_views[0][k], data_range=255, channel_axis=2))
        m = float(np.mean(ss)) if ss else -1e9
        if m > best_s:
            best_s, best_o = m, o
    return best_o, min(P, G - best_o)


# ── one clip ───────────────────────────────────────────────────────────────────
def run_clip(policy, Batch, dataset_dir, ep_idx, start, num_steps, fpc, instruction,
             out_dir, fps_out, max_align_off, verbose):
    import imageio.v2 as iio

    n_needed = 1 + num_steps * fpc
    gt_buffers = read_gt_window(dataset_dir, ep_idx, CAMERA_KEYS, start, n_needed)

    all_preds, latent_video = [], None
    for step in range(num_steps):
        obs = build_obs(gt_buffers, step, fpc, instruction)
        batch = Batch(obs=obs)
        with torch.no_grad():
            # NOTE: do NOT pass video_only=True -> sim_policy then skips setting
            # batch.act but still references it in the de-batch step (AttributeError).
            # The validated path (scripts/test_generate_video.py) omits it; actions
            # are computed but unused here.
            _, video_pred = policy.lazy_joint_forward_causal(
                batch, latent_video=latent_video
            )
        if verbose and step == 0:
            log.info(f"[probe] video_pred shape={tuple(video_pred.shape)} dtype={video_pred.dtype}")
        all_preds.append(video_pred)
        latent_video = video_pred

    pred_frames = decode_latents(policy, all_preds)  # (P, Hp, Wp, 3)
    P, Hp, Wp, _ = pred_frames.shape
    if verbose:
        log.info(f"[probe] decoded pred_frames={pred_frames.shape}")

    # The model packs views into a 2x2 canvas (DreamTransform._prepare_video):
    #   top-left = view0 (aria), bottom-left = view1 (oakd), right column = black
    #   padding (only filled when a 3rd view exists, e.g. AgiBot wrist cam).
    # So split into quadrants, NOT left/right halves.
    hh, ww = Hp // 2, Wp // 2
    pred_views = {
        CONCAT_ORDER[0]: pred_frames[:, :hh, :ww, :],        # top-left  = aria
        CONCAT_ORDER[1]: pred_frames[:, hh:hh * 2, :ww, :],  # bottom-left = oakd
    }
    vh, vw = hh, ww

    # GT views resized to per-view predicted size; GT window starts at `start`
    gt_views = {}
    for cam in CONCAT_ORDER:
        ck = f"video.{cam}"
        src = gt_buffers[ck]  # (n_needed,480,640,3) starting at `start`
        gt_views[cam] = np.stack([resize_to(f, vh, vw) for f in src], axis=0)

    # temporal alignment (shared offset across views), search on first view
    off, n = best_offset(
        [gt_views[CONCAT_ORDER[0]]], [pred_views[CONCAT_ORDER[0]]], max_align_off
    )

    clip_tag = f"ep{ep_idx:06d}_s{start:05d}"
    per_cam = {}
    for cam in CONCAT_ORDER:
        gt = gt_views[cam][off:off + n]
        pr = pred_views[cam][:n]
        fr = [frame_metrics(gt[i], pr[i]) for i in range(n)]
        per_cam[cam] = dict(
            n_frames=n,
            psnr=[f["psnr"] for f in fr],
            ssim=[f["ssim"] for f in fr],
            mse=[f["mse"] for f in fr],
            psnr_mean=float(np.mean([f["psnr"] for f in fr])),
            ssim_mean=float(np.mean([f["ssim"] for f in fr])),
            mse_mean=float(np.mean([f["mse"] for f in fr])),
        )
        # side-by-side [GT | PRED] mp4 for visual inspection
        sbs = np.concatenate([gt, pr], axis=2)  # along width
        sbs_path = os.path.join(out_dir, f"{clip_tag}_{cam}_GTvsPRED.mp4")
        iio.mimsave(sbs_path, list(sbs), fps=fps_out, codec="libx264")

    meta = dict(
        episode=ep_idx, start=start, num_steps=num_steps, frames_per_chunk=fpc,
        instruction=instruction, pred_resolution=[int(Hp), int(Wp)],
        view_resolution=[int(vh), int(vw)], align_offset=int(off), n_eval=int(n),
        cameras=per_cam,
    )
    with open(os.path.join(out_dir, f"{clip_tag}_metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info(
        f"{clip_tag}: off={off} n={n} | "
        + " | ".join(
            f"{cam}: PSNR={per_cam[cam]['psnr_mean']:.2f} SSIM={per_cam[cam]['ssim_mean']:.3f}"
            for cam in CONCAT_ORDER
        )
    )
    return meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--embodiment", default="franka_orca_bimanual")
    p.add_argument("--episodes", default="2,5,32,73,147",
                   help="comma-separated episode indices")
    p.add_argument("--starts", default="10,800,1600",
                   help="comma-separated start frames per episode")
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--frames-per-chunk", type=int, default=4)
    p.add_argument("--instruction", default="bag the groceries")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fps-out", type=float, default=10.0)
    p.add_argument("--max-align-off", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    # env (mirror test_generate_video.py); ATTENTION_BACKEND left to caller/env
    os.environ.setdefault("TOKENIZER_DIR",
                          "/cluster/scratch/rjiang/dreamzero/checkpoints/umt5-xxl")
    os.environ.setdefault("ATTENTION_BACKEND", "FA2")
    torch._dynamo.config.recompile_limit = 800

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12357")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{rank}")

    from torch.distributed.device_mesh import init_device_mesh
    from tianshou.data import Batch
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

    log.info(f"Loading model from {args.model_path} ...")
    device_mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("ip",))
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(args.embodiment),
        model_path=args.model_path,
        device=device,
        device_mesh=device_mesh,
    )
    log.info("Model loaded.")

    os.makedirs(args.out_dir, exist_ok=True)
    episodes = [int(x) for x in args.episodes.split(",") if x != ""]
    starts = [int(x) for x in args.starts.split(",") if x != ""]

    # episode lengths to skip overflowing starts
    lengths = {}
    with open(os.path.join(args.dataset_dir, "meta", "episodes.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            lengths[d["episode_index"]] = d["length"]

    n_needed = 1 + args.num_steps * args.frames_per_chunk
    all_meta, verbose = [], True
    for ep in episodes:
        for st in starts:
            if ep in lengths and st + n_needed > lengths[ep]:
                log.warning(f"skip ep{ep} start{st}: needs {n_needed}, len={lengths[ep]}")
                continue
            try:
                m = run_clip(policy, Batch, args.dataset_dir, ep, st, args.num_steps,
                             args.frames_per_chunk, args.instruction, args.out_dir,
                             args.fps_out, args.max_align_off, verbose)
                all_meta.append(m)
            except Exception as e:
                log.exception(f"clip ep{ep} start{st} FAILED: {e}")
            verbose = False

    with open(os.path.join(args.out_dir, "all_clips.json"), "w") as f:
        json.dump(all_meta, f, indent=2)
    log.info(f"Done. {len(all_meta)} clips -> {args.out_dir}")


if __name__ == "__main__":
    main()
