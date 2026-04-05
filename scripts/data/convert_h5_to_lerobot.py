"""
Convert HDF5 robot episodes to LeRobot v2 format for DreamZero training.

Converts a folder of HDF5 files (one per episode) into the LeRobot v2 directory
structure: parquet files for state/action data, MP4 videos for camera feeds,
and the required meta/info.json.

Designed for dual Franka + Orca hand setup but configurable for other embodiments.

HDF5 expected structure (per file):
    actions_arm_left:          (T, 7)   float64
    actions_arm_right:         (T, 7)   float64
    actions_hand_left:         (T, 17)  float64
    actions_hand_right:        (T, 17)  float64
    observations/qpos_arm_left:   (T, 7)   float64
    observations/qpos_arm_right:  (T, 7)   float64
    observations/qpos_hand_left:  (T, 17)  float64
    observations/qpos_hand_right: (T, 17)  float64
    observations/images/<cam_name>/color: (T, H, W, 3) uint8

Output:
    output_dir/
    ├── data/chunk-000/
    │   ├── episode_000000.parquet
    │   └── ...
    ├── videos/chunk-000/
    │   ├── observation.images.<cam>/
    │   │   ├── episode_000000.mp4
    │   │   └── ...
    │   └── ...
    └── meta/
        └── info.json

Usage:
    python scripts/data/convert_h5_to_lerobot.py \
        --input-dir ./data/raw_h5 \
        --output-dir ./data/franka_orca_lerobot \
        --fps 50 \
        --task "bimanual manipulation"
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from functools import partial
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Robot configuration — edit these if your HDF5 layout differs
# ---------------------------------------------------------------------------

# Order in which state components are concatenated into observation.state
STATE_KEYS = [
    "observations/qpos_arm_left",    # (T, 7)
    "observations/qpos_arm_right",   # (T, 7)
    "observations/qpos_hand_left",   # (T, 17)
    "observations/qpos_hand_right",  # (T, 17)
]

# Order in which action components are concatenated into action
ACTION_KEYS = [
    "actions_arm_left",    # (T, 7)
    "actions_arm_right",   # (T, 7)
    "actions_hand_left",   # (T, 17)
    "actions_hand_right",  # (T, 17)
]

# Camera image datasets (path inside HDF5 → short name for LeRobot)
CAMERA_KEYS = {
    "observations/images/aria_rgb_cam/color": "aria_rgb_cam",
    "observations/images/oakd_front_view/color": "oakd_front_view",
}

CHUNKS_SIZE = 1000  # episodes per chunk directory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_h5_files(input_dir: Path) -> list[Path]:
    """Collect and sort all .h5 files in the input directory."""
    files = sorted(input_dir.glob("*.h5"))
    if not files:
        files = sorted(input_dir.glob("**/*.h5"))
    return files


def read_and_concat(h5f: h5py.File, keys: list[str]) -> np.ndarray:
    """Read datasets from HDF5 and concatenate along the last axis."""
    arrays = []
    for key in keys:
        arr = h5f[key][:]
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arrays.append(arr)
    return np.concatenate(arrays, axis=-1)


def encode_video(
    frames: np.ndarray,
    output_path: Path,
    fps: float,
    codec: str = "mp4v",
    target_resolution: tuple[int, int] | None = None,
) -> tuple[int, int, int]:
    """Encode (T, H, W, 3) uint8 RGB frames to an MP4 file.

    Returns (H, W, C) of the written frames (after any resize).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, C = frames.shape
    if target_resolution is not None:
        W, H = target_resolution  # target_resolution is (width, height)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    for t in range(T):
        frame = frames[t]
        if target_resolution is not None and (frame.shape[1] != W or frame.shape[0] != H):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        # Convert RGB → BGR for OpenCV
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    return (H, W, C)


def build_parquet(
    state: np.ndarray,
    action: np.ndarray,
    episode_index: int,
    fps: float,
    task: str,
) -> pd.DataFrame:
    """Build a parquet-compatible DataFrame for one episode."""
    T = state.shape[0]
    timestamps = np.arange(T, dtype=np.float64) / fps

    # Store arrays as lists-of-floats per row (LeRobot v2 convention)
    rows = {
        "observation.state": [state[t].tolist() for t in range(T)],
        "action": [action[t].tolist() for t in range(T)],
        "timestamp": timestamps.tolist(),
        "episode_index": [episode_index] * T,
        "frame_index": list(range(T)),
        "index": list(range(T)),  # global index, will be adjusted later
        "annotation.task": [task] * T,
    }
    return pd.DataFrame(rows)


def build_info_json(
    total_episodes: int,
    fps: float,
    state_dim: int,
    action_dim: int,
    camera_shapes: dict[str, tuple],
) -> dict:
    """Build the meta/info.json structure."""
    features = {
        "observation.state": {
            "dtype": "float64",
            "shape": [state_dim],
            "names": None,
        },
        "action": {
            "dtype": "float64",
            "shape": [action_dim],
            "names": None,
        },
        "annotation.task": {
            "dtype": "string",
            "shape": [1],
            "names": None,
        },
        "timestamp": {
            "dtype": "float64",
            "shape": [1],
            "names": None,
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
    }

    for cam_name, (H, W, C) in camera_shapes.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [H, W, C],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.fps": fps,
                "video.height": H,
                "video.width": W,
                "video.channels": C,
            },
        }

    return {
        "codebase_version": "v2.0",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "fps": fps,
        "total_episodes": total_episodes,
        "total_frames": 0,  # filled after processing
        "chunks_size": CHUNKS_SIZE,
        "features": features,
    }


# ---------------------------------------------------------------------------
# Per-episode worker (for parallel conversion)
# ---------------------------------------------------------------------------

def convert_episode(
    args_tuple: tuple[int, Path],
    output_dir: Path,
    fps: float,
    task: str,
    target_resolution: tuple[int, int] | None = None,
) -> int:
    """Convert a single episode. Returns the number of frames."""
    ep_idx, h5_path = args_tuple
    chunk_idx = ep_idx // CHUNKS_SIZE

    with h5py.File(h5_path, "r") as h5f:
        state = read_and_concat(h5f, STATE_KEYS)
        action = read_and_concat(h5f, ACTION_KEYS)
        T = state.shape[0]

        # --- Parquet (global index fixed up later) ---
        df = build_parquet(state, action, ep_idx, fps, task)
        parquet_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = parquet_dir / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(parquet_path, index=False)

        # --- Videos ---
        for h5_key, cam_name in CAMERA_KEYS.items():
            frames = h5f[h5_key][:]
            video_dir = (
                output_dir / "videos" / f"chunk-{chunk_idx:03d}"
                / f"observation.images.{cam_name}"
            )
            video_path = video_dir / f"episode_{ep_idx:06d}.mp4"
            encode_video(frames, video_path, fps, target_resolution=target_resolution)

    return T


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    input_dir: Path,
    output_dir: Path,
    fps: float,
    task: str,
    max_episodes: int | None = None,
    num_workers: int | None = None,
    target_resolution: tuple[int, int] | None = None,
) -> None:
    h5_files = get_h5_files(input_dir)
    if not h5_files:
        log.error("No .h5 files found in %s", input_dir)
        sys.exit(1)

    if max_episodes is not None:
        h5_files = h5_files[:max_episodes]

    log.info("Found %d HDF5 episodes in %s", len(h5_files), input_dir)

    # Probe first file for dimensions and camera info
    with h5py.File(h5_files[0], "r") as h5f:
        state_dim = sum(h5f[k].shape[-1] for k in STATE_KEYS)
        action_dim = sum(h5f[k].shape[-1] for k in ACTION_KEYS)
        camera_shapes = {}
        for h5_key, cam_name in CAMERA_KEYS.items():
            shape = h5f[h5_key].shape  # (T, H, W, C)
            if target_resolution is not None:
                W, H = target_resolution
                camera_shapes[cam_name] = (H, W, shape[3])
            else:
                camera_shapes[cam_name] = (shape[1], shape[2], shape[3])

    log.info("State dim: %d, Action dim: %d", state_dim, action_dim)
    log.info("Cameras: %s", {k: v for k, v in camera_shapes.items()})
    if target_resolution is not None:
        log.info("Target resolution: %dx%d", target_resolution[0], target_resolution[1])

    # Prepare output dirs
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Write info.json early (total_frames updated after conversion)
    info = build_info_json(len(h5_files), fps, state_dim, action_dim, camera_shapes)
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    log.info("Wrote initial info.json")

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, len(h5_files))

    worker_fn = partial(convert_episode, output_dir=output_dir, fps=fps, task=task, target_resolution=target_resolution)
    indexed_files = list(enumerate(h5_files))

    if num_workers <= 1:
        # Sequential fallback
        frame_counts = []
        for item in tqdm(indexed_files, desc="Converting episodes"):
            frame_counts.append(worker_fn(item))
    else:
        log.info("Using %d parallel workers", num_workers)
        with mp.Pool(num_workers) as pool:
            frame_counts = list(tqdm(
                pool.imap(worker_fn, indexed_files),
                total=len(indexed_files),
                desc="Converting episodes",
            ))

    # --- Fix up global index in parquet files ---
    cumulative = 0
    for ep_idx, T in enumerate(frame_counts):
        if cumulative > 0:
            chunk_idx = ep_idx // CHUNKS_SIZE
            parquet_path = output_dir / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
            df = pd.read_parquet(parquet_path)
            df["index"] = df["index"] + cumulative
            df.to_parquet(parquet_path, index=False)
        cumulative += T

    total_frames = cumulative

    # --- Update info.json with final total_frames ---
    info["total_frames"] = total_frames
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    log.info("Conversion complete!")
    log.info("  Output: %s", output_dir)
    log.info("  Episodes: %d", len(h5_files))
    log.info("  Total frames: %d", total_frames)
    log.info("  FPS: %s", fps)
    log.info("  State dim: %d, Action dim: %d", state_dim, action_dim)


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 robot episodes to LeRobot v2 format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing .h5 episode files",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for LeRobot v2 dataset",
    )
    parser.add_argument(
        "--fps", type=float, default=50.0,
        help="Recording frequency in Hz (default: 50)",
    )
    parser.add_argument(
        "--task", type=str, default="bimanual manipulation",
        help="Task annotation string for all episodes (default: 'bimanual manipulation')",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Process at most N episodes (for testing)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of parallel workers (default: number of CPUs)",
    )
    parser.add_argument(
        "--target-resolution", type=str, default=None,
        help="Resize all cameras to WxH (e.g. '640x480'). Required when cameras have different resolutions.",
    )
    args = parser.parse_args()

    target_resolution = None
    if args.target_resolution:
        w, h = args.target_resolution.split("x")
        target_resolution = (int(w), int(h))

    convert(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        fps=args.fps,
        task=args.task,
        max_episodes=args.max_episodes,
        num_workers=args.num_workers,
        target_resolution=target_resolution,
    )


if __name__ == "__main__":
    main()
