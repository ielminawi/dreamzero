"""Minimal franka_orca_bimanual test client for the DreamZero websocket server.

Sends the two-Franka + Orca-hand observation the server expects:
  - observation/exterior_image_0_left  -> video.aria_rgb_cam      (H,W,3 uint8)
  - observation/exterior_image_1_left  -> video.oakd_front_view   (H,W,3 uint8)
  - observation/full_state             -> 48-dim [l_arm7, r_arm7, l_hand17, r_hand17]
  - prompt                             -> language instruction

Usage:
  python test_client_franka_orca.py --host localhost --port 5000 --num-chunks 3
"""
import argparse
import logging
import time
import uuid

import numpy as np

from eval_utils.policy_client import WebsocketClientPolicy

# Raw camera resolution expected by the franka_orca transforms (W=640, H=480).
H, W = 480, 640


def make_obs(prompt: str, session_id: str, first: bool) -> dict:
    n = 1 if first else 4  # first call: 1 frame, then FRAMES_PER_CHUNK=4
    obs = {
        "observation/exterior_image_0_left": np.full((n, H, W, 3), 128, dtype=np.uint8),
        "observation/exterior_image_1_left": np.full((n, H, W, 3), 128, dtype=np.uint8),
        "observation/full_state": np.zeros(48, dtype=np.float32),
        "prompt": prompt,
        "session_id": session_id,
    }
    return obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--num-chunks", type=int, default=3)
    p.add_argument("--prompt", default="pick up the object")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    log.info(f"Connecting to {args.host}:{args.port} ...")
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    meta = client.get_server_metadata()
    log.info(f"Server metadata: {meta}")

    session_id = str(uuid.uuid4())
    for i in range(args.num_chunks):
        obs = make_obs(args.prompt, session_id, first=(i == 0))
        t0 = time.time()
        actions = client.infer(obs)
        dt = time.time() - t0
        arr = np.asarray(actions["actions"] if isinstance(actions, dict) and "actions" in actions else actions)
        log.info(f"[{i+1}/{args.num_chunks}] action shape={arr.shape} dt={dt:.2f}s "
                 f"min={arr.min():.3f} max={arr.max():.3f}")

    client.reset({})
    log.info("Done.")


if __name__ == "__main__":
    main()
