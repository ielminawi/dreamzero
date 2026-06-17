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

import os
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
parser.add_argument(
    "--joint-actions",
    action="store_true",
    default=(os.environ.get("JOINT_ACTIONS", "0") == "1"),
    help="JOINT-SPACE policy: arm dims [0:14] of state/action are Panda JOINT ANGLES "
         "(7 per arm), not EE poses. Bypasses FK on obs and IK on actions — arm joints "
         "are read/applied directly, exactly like the hands. Default (off) keeps the "
         "EE+IK path. Also enabled by env JOINT_ACTIONS=1.",
)
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
from eval_utils import ee_ik


class DreamZeroBimanualClient:
    """Client that sends bimanual observations to DreamZero and receives 48-dim actions.

    The policy's arm segments are EE POSES, not joint angles (docs/
    EE_POSE_AND_IK_PIPELINE.md): dims [0:7]/[7:14] of both state and action are
    [x,y,z,qx,qy,qz,qw] flange poses in the arm base frame (XYZW quat). So:
      - obs side: arm proprioception = FK(sim joints) -> EE pose in the dataset
        convention (ee_ik.fk_pose7), NOT the sim joint angles;
      - action side: each received chunk's arm poses are IK'd (ee_ik.solve_chunk,
        warm-started from the current sim joints) into Panda joint targets before
        being applied. Hands (17-dim per side) are real joint angles, passed through.

    Observation format sent to server:
        - observation/full_state: (48,) float64
            [left_arm EE pose(7), right_arm EE pose(7), left_hand(17), right_hand(17)]
        - prompt: str
        - session_id: str

    Action format received:
        - actions: (N, 48) float32, same layout (arm segments are EE poses)
    """

    # Inference image resolution (matches training: 640x480 W×H for aria_rgb_cam)
    INFER_WIDTH = 640
    INFER_HEIGHT = 480

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int = 24,
        joint_actions: bool = False,
    ) -> None:
        self.client = WebsocketClientPolicy(remote_host, remote_port)
        self.open_loop_horizon = open_loop_horizon
        # JOINT-SPACE mode: arm dims [0:14] are Panda joint angles, not EE poses.
        # Obs proprio uses sim joints directly (no FK); action arm dims are applied
        # directly as joint targets (no IK) — exactly like the hands.
        self.joint_actions = joint_actions
        if self.joint_actions:
            print("[mode] JOINT-SPACE policy: bypassing FK on obs and IK on actions "
                  "(arm dims [0:14] = Panda joint angles)", flush=True)
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.joint_chunk = None
        self.session_id = str(uuid.uuid4())
        self._bg = None
        self._bg_inited = False
        self._new_traj()

    def _new_traj(self):
        self.traj = {k: [] for k in (
            "ee_cmd", "q_cmd", "q_meas", "ee_meas", "hand_cmd", "hand_meas",
            "ik_pos_err", "ik_ori_err")}

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.joint_chunk = None
        self.session_id = str(uuid.uuid4())
        self._new_traj()

    def dump_traj(self, path: str):
        """Save the per-step trajectory log (EE/joint commanded vs measured) as npz."""
        arrs = {k: np.asarray(v) for k, v in self.traj.items() if len(v)}
        np.savez(path, **arrs)
        # quick tracking summary: command at t vs measurement at t+1
        if len(self.traj["ee_cmd"]) > 1:
            cmd = np.asarray(self.traj["ee_cmd"])[:-1]
            meas = np.asarray(self.traj["ee_meas"])[1:]
            if self.joint_actions:
                # arm slots [0:14] are joint angles (7/arm); report per-arm joint tracking
                for side, s in (("L", 0), ("R", 7)):
                    jerr = np.degrees(np.abs(cmd[:, s:s + 7] - meas[:, s:s + 7]))
                    print(f"[traj] {side} arm JOINT tracking |err| mean {jerr.mean():.2f}deg  "
                          f"p95 {np.percentile(jerr, 95):.2f}deg  max {jerr.max():.2f}deg", flush=True)
            else:
                for side, s in (("L", 0), ("R", 7)):
                    perr = np.linalg.norm(cmd[:, s:s + 3] - meas[:, s:s + 3], axis=1) * 1000
                    print(f"[traj] {side} arm EE tracking |pos err| mean {perr.mean():.1f}mm  "
                          f"p95 {np.percentile(perr, 95):.1f}mm  max {perr.max():.1f}mm", flush=True)
                ik_pe = np.asarray(self.traj["ik_pos_err"])
                if ik_pe.size:
                    print(f"[traj] IK residual max {ik_pe.max() * 1000:.2f}mm "
                          f"(policy-pose reachability; large => policy asked for unreachable poses)",
                          flush=True)
        print(f"[traj] wrote {path}", flush=True)

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to inference resolution with padding."""
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return cv2.resize(img, (self.INFER_WIDTH, self.INFER_HEIGHT))

    def _maybe_bg(self, img: np.ndarray) -> np.ndarray:
        """Apply the SAME background removal used in preprocessing, so train==eval.

        Gated by env DZ_BG_REMOVAL (none|segformer); must match how the training
        dataset was converted (bg_removal.BackgroundRemover). Off by default.
        """
        method = os.environ.get("DZ_BG_REMOVAL", "none")
        if method == "none":
            return img
        if not self._bg_inited:
            self._bg_inited = True
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            import bg_removal
            self._bg = bg_removal.BackgroundRemover(
                model_name=os.environ.get("DZ_BG_MODEL", bg_removal.DEFAULT_MODEL),
                fill=int(os.environ.get("DZ_BG_FILL", bg_removal.DEFAULT_FILL)),
            )
            print(f"[bg] inference bg removal ON ({method}, fill={self._bg.fill})", flush=True)
        return self._bg.apply_one(img)

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
        q_now = curr_obs["sim_joints"]  # (14,) raw sim arm joints [left7, right7]

        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            self.actions_from_chunk_completed = 0

            # Build request in the format expected by the inference server
            # Map to the camera keys the server expects
            aria_resized = self._maybe_bg(self._resize_image(curr_obs["aria_rgb_cam"]))
            oakd_resized = self._maybe_bg(self._resize_image(curr_obs["oakd_front_view"]))

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
            self.pred_action_chunk = np.asarray(actions, dtype=np.float64)

            if self.joint_actions:
                # JOINT-SPACE: arm dims [0:14] are already Panda joint targets — apply
                # them directly, no IK. (Hands handled the same way below.)
                self.joint_chunk = self.pred_action_chunk[:, 0:14].copy()  # (N, 14)
                self.chunk_ik_pos_err = None
                self.chunk_ik_ori_err = None
                lo = ee_ik.LIMITS[:, 0]
                hi = ee_ik.LIMITS[:, 1]
                jc = self.joint_chunk
                viol = ((jc[:, 0:7] < lo) | (jc[:, 0:7] > hi)).sum() + \
                       ((jc[:, 7:14] < lo) | (jc[:, 7:14] > hi)).sum()
                print(f"[joint] chunk arm-joint range L[{jc[:, 0:7].min():.2f},{jc[:, 0:7].max():.2f}] "
                      f"R[{jc[:, 7:14].min():.2f},{jc[:, 7:14].max():.2f}] "
                      f"limit-violations={int(viol)} nan={int(np.isnan(jc).sum())}", flush=True)
            else:
                # The arm segments of the chunk are EE poses -> IK them into Panda joint
                # targets, warm-started from the CURRENT sim joints so the trajectory is
                # continuous from where the robot actually is.
                ql, pel, oel = ee_ik.solve_chunk(self.pred_action_chunk[:, 0:7], q_now[0:7])
                qr, per, oer = ee_ik.solve_chunk(self.pred_action_chunk[:, 7:14], q_now[7:14])
                self.joint_chunk = np.concatenate([ql, qr], axis=1)  # (N, 14)
                self.chunk_ik_pos_err = np.stack([pel, per], axis=1)  # (N, 2)
                self.chunk_ik_ori_err = np.stack([oel, oer], axis=1)
                print(f"[ik] chunk solved: pos resid max {self.chunk_ik_pos_err.max()*1000:.2f}mm "
                      f"ori max {np.degrees(self.chunk_ik_ori_err.max()):.2f}deg", flush=True)

        i = self.actions_from_chunk_completed
        ee_action = np.asarray(self.pred_action_chunk[i])
        joints = self.joint_chunk[i]
        action = np.concatenate([joints, ee_action[14:48]]).astype(np.float64)
        self.actions_from_chunk_completed += 1

        # per-step trajectory log (cmd at t, measurement at t)
        self.traj["ee_cmd"].append(ee_action[0:14].copy())
        self.traj["q_cmd"].append(joints.copy())
        self.traj["q_meas"].append(q_now.copy())
        self.traj["ee_meas"].append(curr_obs["state"][0:14].copy())
        self.traj["hand_cmd"].append(ee_action[14:48].copy())
        self.traj["hand_meas"].append(curr_obs["state"][14:48].copy())
        if not self.joint_actions:
            self.traj["ik_pos_err"].append(self.chunk_ik_pos_err[i].copy())
            self.traj["ik_ori_err"].append(self.chunk_ik_ori_err[i].copy())

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

        # Proprioception: concatenate to 48-dim state vector.
        left_arm = policy["left_arm_joint_pos"][0].clone().detach().cpu().numpy()    # (7,) sim joints
        right_arm = policy["right_arm_joint_pos"][0].clone().detach().cpu().numpy()   # (7,) sim joints
        left_hand = policy["left_hand_joint_pos"][0].clone().detach().cpu().numpy()   # (17,)
        right_hand = policy["right_hand_joint_pos"][0].clone().detach().cpu().numpy() # (17,)

        if self.joint_actions:
            # JOINT-SPACE: arm proprio = sim joint angles directly (no FK), matching the
            # joint-space training distribution. Hands are joint angles in both modes.
            state = np.concatenate([left_arm, right_arm, left_hand, right_hand])  # (48,)
        else:
            # The policy was trained with EE POSES in the arm slots, not joint angles. Emit
            # the FK-computed flange pose in the exact dataset convention (position in the
            # arm base frame, XYZW quat, qx>0 hemisphere) so proprioception is in-distribution.
            left_ee = ee_ik.fk_pose7(left_arm)
            right_ee = ee_ik.fk_pose7(right_arm)
            state = np.concatenate([left_ee, right_ee, left_hand, right_hand])  # (48,) dataset conv

        return {
            "aria_rgb_cam": aria_rgb,
            "oakd_front_view": oakd_front,
            "state": state,
            "sim_joints": np.concatenate([left_arm, right_arm]),  # (14,) raw, for IK warm start
        }


def main():
    print("\n=== Creating FrankaOrcaBimanual-v0 environment ===")
    cfg = FrankaOrcaBimanualEnvCfg()
    cfg.sim.device = args.device  # honor --device (Blackwell GPUs: Isaac's bundled torch is cu118
    # with no sm_120 kernels, so run PhysX/state on cpu; RTX camera rendering stays on the GPU)
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
        joint_actions=args.joint_actions,
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

            client.dump_traj(str(video_dir / f"episode_{ep}_traj.npz"))
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
