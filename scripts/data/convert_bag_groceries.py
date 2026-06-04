"""
Convert the `bag_groceries` raw HDF5 dataset to LeRobot v2 for franka_orca_bimanual
training, with optional background removal and SLURM-array sharding.

Why a dedicated script (vs convert_h5_to_lerobot.py): background removal runs a GPU
segmentation model, so we process episodes sequentially in one GPU process per array
task instead of a CPU multiprocessing Pool. Episodes are split across `--num-shards`
tasks (round-robin); a final `--finalize` pass (run once, after all shards) writes
meta/info.json and fixes the global frame index.

Pipeline per camera frame:  raw -> resize to --target-resolution (640x480) ->
[optional background removal at that resolution] -> encode mp4. The SAME resize+bg
transform is applied to sim frames at inference (run_sim_eval_bimanual) so train==eval.

HDF5 layout (bag_groceries):
    actions_arm_left/right (T,7), actions_hand_left/right (T,17)
    observations/qpos_arm_left/right (T,7), observations/qpos_hand_left/right (T,17)
    observations/images/{aria_rgb_cam,oakd_front_view}/color (T,H,W,3) uint8

state/action concat order -> indices:
    left_arm[0:7] right_arm[7:14] left_hand[14:31] right_hand[31:48]   (dim 48)

Example (one array task):
    python scripts/data/convert_bag_groceries.py \
        --input-dir /cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries \
        --output-dir /cluster/scratch/rjiang/dreamzero/data/franka_orca_bag_groceries_lerobot \
        --fps 50 --target-resolution 640x480 --bg-removal segformer \
        --num-shards 10 --shard-index 3 --task "pack the groceries into the bag"

Finalize (once, after all shards finish):
    python scripts/data/convert_bag_groceries.py --input-dir ... --output-dir ... \
        --fps 50 --target-resolution 640x480 --finalize
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

# repo root on path so `import bg_removal` works when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("convert_bag_groceries")

STATE_KEYS = [
    "observations/qpos_arm_left",    # (T,7)
    "observations/qpos_arm_right",   # (T,7)
    "observations/qpos_hand_left",   # (T,17)
    "observations/qpos_hand_right",  # (T,17)
]
ACTION_KEYS = [
    "actions_arm_left",    # (T,7)
    "actions_arm_right",   # (T,7)
    "actions_hand_left",   # (T,17)
    "actions_hand_right",  # (T,17)
]
CAMERA_KEYS = {
    "observations/images/aria_rgb_cam/color": "aria_rgb_cam",
    "observations/images/oakd_front_view/color": "oakd_front_view",
}
CHUNKS_SIZE = 1000
SIDECAR_DIR = ".sidecars"


def get_h5_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.h5"))
    if not files:
        files = sorted(input_dir.glob("**/*.h5"))
    return files


def read_and_concat(h5f: h5py.File, keys: list[str]) -> np.ndarray:
    arrays = []
    for key in keys:
        arr = h5f[key][:]
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arrays.append(np.asarray(arr, dtype=np.float64))
    return np.concatenate(arrays, axis=-1)


def encode_video(frames: np.ndarray, output_path: Path, fps: float) -> None:
    """Encode (T,H,W,3) uint8 RGB to mp4 (mp4v). Frames must already be target size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, _ = frames.shape
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {output_path}")
    for t in range(T):
        writer.write(cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR))
    writer.release()


def resize_frames(frames: np.ndarray, target_wh: tuple[int, int] | None) -> np.ndarray:
    if target_wh is None:
        return frames
    W, H = target_wh
    if frames.shape[2] == W and frames.shape[1] == H:
        return frames
    out = np.empty((frames.shape[0], H, W, 3), dtype=np.uint8)
    for t in range(frames.shape[0]):
        out[t] = cv2.resize(frames[t], (W, H), interpolation=cv2.INTER_AREA)
    return out


def build_parquet(state, action, ep_idx, fps, task) -> pd.DataFrame:
    T = state.shape[0]
    return pd.DataFrame({
        "observation.state": [state[t].tolist() for t in range(T)],
        "action": [action[t].tolist() for t in range(T)],
        "timestamp": (np.arange(T, dtype=np.float64) / fps).tolist(),
        "episode_index": [ep_idx] * T,
        "frame_index": list(range(T)),
        "index": list(range(T)),  # global offset applied in finalize
        "task_index": [0] * T,
        "annotation.task": [task] * T,
    })


def convert_episode(ep_idx, h5_path, output_dir, fps, task, target_wh, remover) -> int:
    chunk_idx = ep_idx // CHUNKS_SIZE
    with h5py.File(h5_path, "r") as h5f:
        state = read_and_concat(h5f, STATE_KEYS)
        action = read_and_concat(h5f, ACTION_KEYS)
        T = state.shape[0]

        df = build_parquet(state, action, ep_idx, fps, task)
        pq_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
        pq_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq_dir / f"episode_{ep_idx:06d}.parquet", index=False)

        for h5_key, cam in CAMERA_KEYS.items():
            frames = h5f[h5_key][:]                         # (T,Hn,Wn,3) uint8
            frames = resize_frames(frames, target_wh)        # -> 640x480
            if remover is not None:
                frames = remover.apply(frames)               # bg -> constant fill
            vdir = output_dir / "videos" / f"chunk-{chunk_idx:03d}" / f"observation.images.{cam}"
            encode_video(frames, vdir / f"episode_{ep_idx:06d}.mp4", fps)
            del frames

    sc = output_dir / "meta" / SIDECAR_DIR
    sc.mkdir(parents=True, exist_ok=True)
    (sc / f"episode_{ep_idx:06d}.json").write_text(json.dumps({"episode_index": ep_idx, "length": int(T)}))
    return T


def finalize(output_dir: Path, h5_files: list[Path], fps: float, target_wh, task: str) -> None:
    n = len(h5_files)
    sc = output_dir / "meta" / SIDECAR_DIR
    lengths = {}
    missing = []
    for i in range(n):
        p = sc / f"episode_{i:06d}.json"
        if not p.exists():
            missing.append(i)
            continue
        lengths[i] = json.loads(p.read_text())["length"]
    if missing:
        raise SystemExit(f"finalize: {len(missing)} episodes missing sidecars (e.g. {missing[:8]}); "
                         f"rerun the failed shard(s) before finalize.")

    # global index fixup
    cumulative = 0
    for ep_idx in range(n):
        T = lengths[ep_idx]
        if cumulative > 0:
            chunk_idx = ep_idx // CHUNKS_SIZE
            pp = output_dir / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
            df = pd.read_parquet(pp)
            if int(df["index"].iloc[0]) == 0:  # not already offset (idempotent-ish)
                df["index"] = np.arange(len(df)) + cumulative
                df.to_parquet(pp, index=False)
        cumulative += T
    total_frames = cumulative

    # camera shapes after resize
    W, H = target_wh if target_wh else (640, 480)
    state_dim = sum({"arm": 7, "hand": 17}["arm" if "arm" in k else "hand"] for k in STATE_KEYS)
    action_dim = state_dim
    features = {
        "observation.state": {"dtype": "float64", "shape": [state_dim], "names": None},
        "action": {"dtype": "float64", "shape": [action_dim], "names": None},
        "annotation.task": {"dtype": "string", "shape": [1], "names": None},
        "timestamp": {"dtype": "float64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for cam in CAMERA_KEYS.values():
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channels"],
            "video_info": {"video.fps": fps, "video.height": H, "video.width": W, "video.channels": 3},
        }
    info = {
        "codebase_version": "v2.0",
        "robot_type": "franka_orca_bimanual",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "fps": fps,
        "total_episodes": n,
        "total_frames": total_frames,
        "total_videos": n * len(CAMERA_KEYS),
        "total_tasks": 1,
        "chunks_size": CHUNKS_SIZE,
        "splits": {"train": f"0:{n}"},
        "features": features,
    }
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    log.info("finalize: %d episodes, %d frames, state/action dim %d, cams %s",
             n, total_frames, state_dim, list(CAMERA_KEYS.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--task", type=str, default="pack the groceries into the bag")
    ap.add_argument("--target-resolution", type=str, default="640x480", help="WxH; '' to keep native")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--bg-removal", choices=["none", "segformer", "mask2former"], default="mask2former")
    ap.add_argument("--bg-model", type=str, default=None)  # auto per backend
    ap.add_argument("--bg-fill", type=int, default=0)       # black
    ap.add_argument("--bg-batch-size", type=int, default=None)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_wh = None
    if args.target_resolution:
        w, h = args.target_resolution.lower().split("x")
        target_wh = (int(w), int(h))

    h5_files = get_h5_files(input_dir)
    if args.max_episodes is not None:
        h5_files = h5_files[: args.max_episodes]
    if not h5_files:
        log.error("No .h5 files in %s", input_dir)
        sys.exit(1)

    if args.finalize:
        finalize(output_dir, h5_files, args.fps, target_wh, args.task)
        return

    my_eps = [i for i in range(len(h5_files)) if i % args.num_shards == args.shard_index]
    log.info("shard %d/%d -> %d episodes (of %d). bg=%s target=%s",
             args.shard_index, args.num_shards, len(my_eps), len(h5_files),
             args.bg_removal, target_wh)

    remover = None
    if args.bg_removal != "none":
        import bg_removal
        remover = bg_removal.BackgroundRemover(backend=args.bg_removal, model_name=args.bg_model,
                                               fill=args.bg_fill, batch_size=args.bg_batch_size)
        log.info("BackgroundRemover ready (backend=%s model=%s device=%s fill=%d)",
                 args.bg_removal, remover.model_name, remover.device, remover.fill)

    for n_done, ep_idx in enumerate(my_eps):
        try:
            T = convert_episode(ep_idx, h5_files[ep_idx], output_dir, args.fps,
                                args.task, target_wh, remover)
            log.info("[%d/%d] ep %d (%s): %d frames", n_done + 1, len(my_eps),
                     ep_idx, h5_files[ep_idx].name, T)
        except Exception as e:  # noqa: BLE001 - keep shard going, finalize will flag gaps
            log.error("ep %d (%s) FAILED: %r", ep_idx, h5_files[ep_idx].name, e)

    log.info("shard %d done (%d episodes)", args.shard_index, len(my_eps))


if __name__ == "__main__":
    main()
