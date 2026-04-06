"""Closed-loop evaluation of DreamZero on bimanual Franka + Orca in Isaac Sim.

Connects to the DreamZero inference server via WebSocket and runs episodes
in the FrankaOrcaBimanual IsaacLab environment.

Usage:
    # Start inference server first (in separate container/terminal):
    torchrun --nproc_per_node=1 socket_test_optimized_AR.py --port 8000 \
        --model-path checkpoints/dreamzero_franka_orca_lora --enable-dit-cache

    # Then run eval:
    python eval_utils/run_sim_eval_bimanual.py --headless --enable_cameras \
        --host localhost --port 8000 \
        --instruction "pick up the cube" --episodes 10
"""

import uuid
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import mediapy
import numpy as np
import torch
from tqdm import tqdm

# Isaac Sim must be launched before any other Isaac imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Bimanual Franka+Orca sim eval")
parser.add_argument("--host", type=str, default="localhost", help="Inference server host")
parser.add_argument("--port", type=int, default=8000, help="Inference server port")
parser.add_argument("--instruction", type=str, default="pick up the object", help="Language task instruction")
parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
parser.add_argument("--open-loop-horizon", type=int, default=24, help="Actions per inference chunk")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now safe to import Isaac/sim_envs
import sys
sys.path.insert(0, "/app")

# Enable URDF importer extension
import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
for ext_name in ["isaacsim.asset.importer.urdf", "omni.importer.urdf", "omni.isaac.urdf"]:
    if ext_manager.set_extension_enabled_immediate(ext_name, True):
        print(f"[INFO] Enabled extension: {ext_name}")
        break

import sim_envs  # noqa: F401 — registers the gym environment
from sim_envs.franka_orca_bimanual_cfg import FrankaOrcaBimanualEnvCfg
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv
from eval_utils.policy_client import WebsocketClientPolicy


class DreamZeroBimanualClient:
    """Client that sends bimanual observations to DreamZero and receives 48-dim actions.

    Observation format sent to server:
        - observation/aria_rgb_cam: (H, W, 3) uint8
        - observation/oakd_front_view: (H, W, 3) uint8
        - observation/state: (48,) float64
            [left_arm(7), right_arm(7), left_hand(17), right_hand(17)]
        - prompt: str
        - session_id: str

    Action format received:
        - actions: (N, 48) float32
            [left_arm(7), right_arm(7), left_hand(17), right_hand(17)]
    """

    # Inference image resolution (matches training: 320x176 W×H)
    INFER_WIDTH = 320
    INFER_HEIGHT = 176

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int = 24,
    ) -> None:
        self.client = WebsocketClientPolicy(remote_host, remote_port)
        self.open_loop_horizon = open_loop_horizon
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to inference resolution with padding."""
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return cv2.resize(img, (self.INFER_WIDTH, self.INFER_HEIGHT))

    def infer(self, obs: dict, instruction: str) -> dict:
        """Run inference, returning the next action and visualization.

        Args:
            obs: Isaac Sim observation dict with keys:
                policy/left_arm_joint_pos, policy/right_arm_joint_pos,
                policy/left_hand_joint_pos, policy/right_hand_joint_pos,
                policy/aria_rgb_cam, policy/oakd_front_view
            instruction: Language task description

        Returns:
            dict with "action" (48,) and "viz" (combined camera view)
        """
        curr_obs = self._extract_observation(obs)

        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            self.actions_from_chunk_completed = 0

            # Build request in the format expected by the inference server
            # Map to the camera keys the server expects
            aria_resized = self._resize_image(curr_obs["aria_rgb_cam"])
            oakd_resized = self._resize_image(curr_obs["oakd_front_view"])

            request_data = {
                # Map 2 cameras to the server's expected keys
                "observation/exterior_image_0_left": aria_resized,
                "observation/exterior_image_1_left": oakd_resized,
                # No wrist camera — send a blank image
                "observation/wrist_image_left": np.zeros(
                    (self.INFER_HEIGHT, self.INFER_WIDTH, 3), dtype=np.uint8
                ),
                # State: concatenate all proprioception into the expected format
                # The server's _convert_observation maps these to state keys
                "observation/joint_position": curr_obs["state"][:14].astype(np.float64),
                "observation/gripper_position": np.zeros((1,), dtype=np.float64),
                "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
                # Full state as additional field for 48-dim embodiments
                "observation/full_state": curr_obs["state"].astype(np.float64),
                "prompt": instruction,
                "session_id": self.session_id,
            }

            result = self.client.infer(request_data)
            actions = result["actions"] if isinstance(result, dict) else result
            assert len(actions.shape) == 2, f"Expected 2D array, got shape {actions.shape}"
            self.pred_action_chunk = actions

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # Build visualization
        aria_viz = cv2.resize(curr_obs["aria_rgb_cam"], (320, 240))
        oakd_viz = cv2.resize(curr_obs["oakd_front_view"], (320, 240))
        viz = np.concatenate([aria_viz, oakd_viz], axis=1)

        return {"action": action, "viz": viz}

    def _extract_observation(self, obs_dict: dict) -> dict:
        """Extract numpy arrays from Isaac Sim observation dict."""
        policy = obs_dict["policy"]

        # Camera images: (N, H, W, 3) -> (H, W, 3) for env 0
        aria_rgb = policy["aria_rgb_cam"][0].clone().detach().cpu().numpy()
        oakd_front = policy["oakd_front_view"][0].clone().detach().cpu().numpy()

        # Proprioception: concatenate to 48-dim state vector
        left_arm = policy["left_arm_joint_pos"][0].clone().detach().cpu().numpy()    # (7,)
        right_arm = policy["right_arm_joint_pos"][0].clone().detach().cpu().numpy()   # (7,)
        left_hand = policy["left_hand_joint_pos"][0].clone().detach().cpu().numpy()   # (17,)
        right_hand = policy["right_hand_joint_pos"][0].clone().detach().cpu().numpy() # (17,)

        state = np.concatenate([left_arm, right_arm, left_hand, right_hand])  # (48,)

        return {
            "aria_rgb_cam": aria_rgb,
            "oakd_front_view": oakd_front,
            "state": state,
        }


def main():
    print("\n=== Creating FrankaOrcaBimanual-v0 environment ===")
    cfg = FrankaOrcaBimanualEnvCfg()
    env = FrankaOrcaBimanualEnv(cfg)

    # Double reset (first loads assets, second stabilises materials)
    print("Reset 1/2 (loading assets)...")
    obs, _ = env.reset()
    print("Reset 2/2 (stabilising)...")
    obs, _ = env.reset()

    # Connect to DreamZero inference server
    print(f"\nConnecting to inference server at {args.host}:{args.port}...")
    client = DreamZeroBimanualClient(
        remote_host=args.host,
        remote_port=args.port,
        open_loop_horizon=args.open_loop_horizon,
    )

    # Output directory for videos
    video_dir = Path("/output") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)

    max_steps = env.max_episode_length

    with torch.no_grad():
        for ep in range(args.episodes):
            video = []
            for step in tqdm(range(max_steps), desc=f"Episode {ep+1}/{args.episodes}"):
                ret = client.infer(obs, args.instruction)

                if not args.headless:
                    cv2.imshow("Cameras", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

                video.append(ret["viz"])

                # Apply 48-dim action
                action = torch.tensor(ret["action"], dtype=torch.float32, device=env.device).unsqueeze(0)
                obs, _ = env.step(action)

            client.reset()
            video_path = video_dir / f"episode_{ep}.mp4"
            mediapy.write_video(str(video_path), video, fps=50)
            obs, _ = env.reset()

            print(f"Episode {ep+1} saved to {video_path}")

    print(f"\n=== Evaluation complete — {args.episodes} episodes saved to {video_dir} ===")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
