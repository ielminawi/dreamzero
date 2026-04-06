"""IsaacLab environment for dual Franka + Orca hand bimanual manipulation.

Each side is a single 24-DOF articulation (Franka arm + Orca hand):
  joints[0:7]  = panda arm
  joints[7:24] = orca hand

Action space: 48 dims = [left_arm(7), right_arm(7), left_hand(17), right_hand(17)]
State space:  48 dims (same structure)
Cameras: aria_rgb_cam (480x640), oakd_front_view (540x960)
"""

from __future__ import annotations

import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedEnv

from sim_envs.franka_orca_bimanual_cfg import FrankaOrcaBimanualEnvCfg


class FrankaOrcaBimanualEnv(ManagerBasedEnv):
    """Dual Franka + Orca bimanual manipulation environment."""

    cfg: FrankaOrcaBimanualEnvCfg

    # Action dimensions per component
    LEFT_ARM_DIM = 7
    RIGHT_ARM_DIM = 7
    LEFT_HAND_DIM = 17
    RIGHT_HAND_DIM = 17
    TOTAL_ACTION_DIM = 48

    # Joint indices within each 24-DOF articulation
    ARM_SLICE = slice(0, 7)
    HAND_SLICE = slice(7, 24)

    def __init__(self, cfg: FrankaOrcaBimanualEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, **kwargs)
        self.max_episode_length = int(cfg.episode_length_s / cfg.sim.dt)
        self._step_count = 0
        self._relative_actions = True  # Match training config

    def _apply_action(self, action: torch.Tensor) -> None:
        """Apply 48-dim action to both arm+hand articulations.

        Args:
            action: (num_envs, 48) tensor:
                [0:7]   = left arm joint positions
                [7:14]  = right arm joint positions
                [14:31] = left hand joint positions
                [31:48] = right hand joint positions
        """
        left_arm_action = action[:, 0:7]
        right_arm_action = action[:, 7:14]
        left_hand_action = action[:, 14:31]
        right_hand_action = action[:, 31:48]

        # Build per-articulation targets (24-dim each: 7 arm + 17 hand)
        left_data = self.scene["left_arm_hand"].data
        right_data = self.scene["right_arm_hand"].data

        if self._relative_actions:
            left_arm_target = left_data.joint_pos[:, self.ARM_SLICE] + left_arm_action
            left_hand_target = left_data.joint_pos[:, self.HAND_SLICE] + left_hand_action
            right_arm_target = right_data.joint_pos[:, self.ARM_SLICE] + right_arm_action
            right_hand_target = right_data.joint_pos[:, self.HAND_SLICE] + right_hand_action
        else:
            left_arm_target = left_arm_action
            left_hand_target = left_hand_action
            right_arm_target = right_arm_action
            right_hand_target = right_hand_action

        # Concatenate arm + hand targets per articulation
        left_target = torch.cat([left_arm_target, left_hand_target], dim=-1)
        right_target = torch.cat([right_arm_target, right_hand_target], dim=-1)

        self.scene["left_arm_hand"].set_joint_position_target(left_target)
        self.scene["right_arm_hand"].set_joint_position_target(right_target)

    def _get_observations(self) -> dict:
        """Get observations matching training data format."""
        left_data = self.scene["left_arm_hand"].data
        right_data = self.scene["right_arm_hand"].data

        obs = {
            "policy": {
                "left_arm_joint_pos": left_data.joint_pos[:, self.ARM_SLICE],
                "right_arm_joint_pos": right_data.joint_pos[:, self.ARM_SLICE],
                "left_hand_joint_pos": left_data.joint_pos[:, self.HAND_SLICE],
                "right_hand_joint_pos": right_data.joint_pos[:, self.HAND_SLICE],
                "aria_rgb_cam": self.scene["aria_rgb_cam"].data.output["rgb"][..., :3],
                "oakd_front_view": self.scene["oakd_front_view"].data.output["rgb"][..., :3],
            }
        }
        return obs

    def _get_rewards(self) -> torch.Tensor:
        """No reward — this is an open-loop eval environment."""
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Episode termination based on step count."""
        self._step_count += 1
        time_out = self._step_count >= self.max_episode_length
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.full((self.num_envs,), time_out, dtype=torch.bool, device=self.device)
        return terminated, truncated

    def _reset_idx(self, env_ids):
        """Reset environments to initial state."""
        super()._reset_idx(env_ids)
        self._step_count = 0
