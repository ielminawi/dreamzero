"""IsaacLab environments for DreamZero simulation evaluation."""

import gymnasium as gym

gym.register(
    id="FrankaOrcaBimanual-v0",
    entry_point="sim_envs.franka_orca_bimanual_env:FrankaOrcaBimanualEnv",
)
