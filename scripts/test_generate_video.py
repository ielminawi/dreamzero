#!/usr/bin/env python3
"""
Offline video generation test for DreamZero.

Runs the model checkpoint on input video frames (from MP4 files) and generates
predicted future video autoregressively. No robot connection required.

Examples:
    # DROID model with debug images (3 cameras):
    python scripts/test_generate_video.py \
        --model-path checkpoints/DreamZero-DROID \
        --embodiment oxe_droid \
        --video-dir debug_image \
        --instruction "pick up the object" \
        --num-steps 5 \
        --output-dir output/test_gen

    # franka-orca bimanual model (2 cameras):
    python scripts/test_generate_video.py \
        --model-path checkpoints/franka_orca_lora \
        --embodiment franka_orca_bimanual \
        --video-dir /path/to/videos \
        --instruction "grasp the block" \
        --num-steps 4

Expected video files in --video-dir:
    oxe_droid:           exterior_image_1_left.mp4, exterior_image_2_left.mp4, wrist_image_left.mp4
    franka_orca_bimanual: aria_rgb_cam.mp4, oakd_front_view.mp4
"""
import argparse
import datetime
import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist


# ── per-embodiment constants ───────────────────────────────────────────────────

CAMERA_KEYS = {
    "oxe_droid": [
        "video.exterior_image_1_left",
        "video.exterior_image_2_left",
        "video.wrist_image_left",
    ],
    "franka_orca_bimanual": [
        "video.aria_rgb_cam",
        "video.oakd_front_view",
    ],
}

VIDEO_FILENAMES = {
    "oxe_droid": {
        "video.exterior_image_1_left": "exterior_image_1_left.mp4",
        "video.exterior_image_2_left": "exterior_image_2_left.mp4",
        "video.wrist_image_left": "wrist_image_left.mp4",
    },
    "franka_orca_bimanual": {
        "video.aria_rgb_cam": "aria_rgb_cam.mp4",
        "video.oakd_front_view": "oakd_front_view.mp4",
    },
}

STATE_KEYS_DIMS = {
    "oxe_droid": {
        "state.joint_position": 7,
        "state.gripper_position": 1,
    },
    "franka_orca_bimanual": {
        "state.left_arm_joint_pos": 7,
        "state.right_arm_joint_pos": 7,
        "state.left_hand_joint_pos": 17,
        "state.right_hand_joint_pos": 17,
    },
}

LANG_KEYS = {
    "oxe_droid": "annotation.language.language_instruction",
    "franka_orca_bimanual": "annotation.task",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def load_video_frames(video_path: str) -> np.ndarray:
    """Return (T, H, W, C) uint8 array from an MP4 file."""
    import imageio
    reader = imageio.get_reader(video_path)
    frames = [f for f in reader]
    reader.close()
    return np.stack(frames, axis=0)


def build_obs(
    video_buffers: dict,
    step: int,
    frames_per_chunk: int,
    state_keys_dims: dict,
    lang_key: str,
    instruction: str,
) -> dict:
    """Build an observation dict for one inference step.

    Step 0 uses 1 conditioning frame; subsequent steps use frames_per_chunk frames.
    """
    obs: dict = {}

    if step == 0:
        n_frames = 1
        start_idx = 0
    else:
        n_frames = frames_per_chunk
        start_idx = 1 + (step - 1) * frames_per_chunk

    for cam_key, all_frames in video_buffers.items():
        total = len(all_frames)
        if start_idx >= total:
            raise ValueError(
                f"input video '{cam_key}' too short for step {step}: "
                f"need at least 1 frame at index {start_idx}, but only {total} available"
            )
        end_idx = min(start_idx + n_frames, total)
        selected = all_frames[start_idx:end_idx]
        if len(selected) < n_frames:
            pad = np.repeat(selected[-1:], n_frames - len(selected), axis=0)
            selected = np.concatenate([selected, pad], axis=0)
        obs[cam_key] = selected  # (n_frames, H, W, C)

    for key, dim in state_keys_dims.items():
        obs[key] = np.zeros((1, dim), dtype=np.float64)

    obs[lang_key] = instruction
    return obs


def decode_latents(policy, latent_list: list) -> np.ndarray:
    """Decode a list of latent video tensors to pixel frames (T, H, W, C) uint8."""
    from einops import rearrange

    ah = policy.trained_model.action_head
    video_cat = torch.cat(latent_list, dim=2)  # (B, 16, T_total, H//8, W//8)
    with torch.no_grad():
        frames = ah.vae.decode(
            video_cat,
            tiled=ah.tiled,
            tile_size=(ah.tile_size_height, ah.tile_size_width),
            tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
        )  # (B, C, T, H, W)
    frames = rearrange(frames, "B C T H W -> B T H W C")[0]
    frames = ((frames.float() + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
    return frames


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Offline DreamZero video generation test")
    p.add_argument("--model-path", required=True, help="Path to checkpoint directory")
    p.add_argument(
        "--embodiment",
        default="franka_orca_bimanual",
        choices=list(CAMERA_KEYS),
        help="Embodiment type",
    )
    p.add_argument(
        "--video-dir",
        default=None,
        help="Directory with per-camera MP4 files (see script docstring for filenames)",
    )
    p.add_argument("--instruction", default="pick up the object", help="Language instruction")
    p.add_argument(
        "--num-steps",
        type=int,
        default=4,
        help="Number of autoregressive generation steps",
    )
    p.add_argument(
        "--frames-per-chunk",
        type=int,
        default=4,
        help="Input frames per step (after the first step)",
    )
    p.add_argument(
        "--output-dir",
        default="output/test_video_gen",
        help="Directory to write generated MP4 files",
    )
    p.add_argument("--fps", type=float, default=5.0, help="Output video FPS")
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip torch.compile (faster startup, slower inference). "
             "Sets ENABLE_TENSORRT=True internally.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # ── env setup ──────────────────────────────────────────────────────────────
    os.environ.setdefault("TOKENIZER_DIR", "/home/ubuntu/dreamzero/checkpoints/umt5-xxl")
    os.environ.setdefault("ATTENTION_BACKEND", "FA2")
    if args.no_compile:
        os.environ["ENABLE_TENSORRT"] = "True"

    # Increase dynamo recompile limit (needed for autoregressive inference).
    torch._dynamo.config.recompile_limit = 800

    # ── single-GPU distributed init ────────────────────────────────────────────
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    rank = int(os.environ["RANK"])
    # NCCL requires set_device BEFORE init_process_group so each rank binds to its GPU.
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{rank}")
    log.info(f"Using device: {device}")

    # ── load model ─────────────────────────────────────────────────────────────
    from torch.distributed.device_mesh import init_device_mesh
    from tianshou.data import Batch

    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

    log.info(f"Loading model from '{args.model_path}' (embodiment={args.embodiment})...")
    device_mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("ip",))
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(args.embodiment),
        model_path=args.model_path,
        device=device,
        device_mesh=device_mesh,
    )
    log.info("Model loaded.")

    # ── load input video frames ────────────────────────────────────────────────
    import imageio

    camera_keys = CAMERA_KEYS[args.embodiment]
    state_keys_dims = STATE_KEYS_DIMS[args.embodiment]
    lang_key = LANG_KEYS[args.embodiment]
    filename_map = VIDEO_FILENAMES[args.embodiment]

    video_buffers: dict[str, np.ndarray] = {}
    for cam_key in camera_keys:
        if args.video_dir is not None:
            video_path = os.path.join(args.video_dir, filename_map[cam_key])
        else:
            video_path = None

        if video_path is not None and os.path.isfile(video_path):
            log.info(f"  Loading {video_path} ...")
            video_buffers[cam_key] = load_video_frames(video_path)
            log.info(f"    {cam_key}: {video_buffers[cam_key].shape}")
        else:
            if video_path is not None:
                log.warning(f"  {video_path} not found, using gray dummy frames")
            else:
                log.info(f"  No --video-dir given, using gray dummy frames for {cam_key}")
            needed = 1 + args.num_steps * args.frames_per_chunk
            video_buffers[cam_key] = np.full((needed, 176, 320, 3), 128, dtype=np.uint8)

    # ── autoregressive inference ───────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    all_video_preds: list[torch.Tensor] = []
    latent_video: torch.Tensor | None = None

    for step in range(args.num_steps):
        log.info(f"Step {step + 1}/{args.num_steps} ...")
        obs = build_obs(
            video_buffers,
            step=step,
            frames_per_chunk=args.frames_per_chunk,
            state_keys_dims=state_keys_dims,
            lang_key=lang_key,
            instruction=args.instruction,
        )
        batch = Batch(obs=obs)

        with torch.no_grad():
            result_batch, video_pred = policy.lazy_joint_forward_causal(
                batch, latent_video=latent_video
            )

        all_video_preds.append(video_pred)
        latent_video = video_pred

    # ── decode and save generated video ───────────────────────────────────────
    log.info("Decoding generated video...")
    gen_frames = decode_latents(policy, all_video_preds)

    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    gen_path = os.path.join(
        args.output_dir,
        f"{args.embodiment}_{timestamp}_{args.num_steps}steps_gen.mp4",
    )
    imageio.mimsave(gen_path, list(gen_frames), fps=args.fps, codec="libx264")
    log.info(f"Saved {len(gen_frames)} generated frames → {gen_path}")

    # ── save input reference videos for comparison ─────────────────────────────
    total_input_frames = 1 + args.num_steps * args.frames_per_chunk
    for cam_key, cam_frames in video_buffers.items():
        ref_path = os.path.join(
            args.output_dir,
            f"{cam_key.replace('.', '_')}_{timestamp}_input.mp4",
        )
        n = min(total_input_frames, len(cam_frames))
        imageio.mimsave(ref_path, list(cam_frames[:n]), fps=args.fps, codec="libx264")
        log.info(f"Saved input reference ({n} frames) → {ref_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
