# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
    success_fraction: float = 0.8,
    ema_alpha: float = 0.1,
    step_size: float = 0.1,
    min_update_interval_episodes: float = 1.0,
) -> None:
    """command_levels_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Initialize the curriculum only once. The env is reset multiple times before the
    # first step, so relying on common_step_counter == 0 reapplies the 0.1 multiplier.
    if not getattr(env, "_command_curriculum_initialized", False):
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]
        env._command_curriculum_tracking_ema = None
        env._command_curriculum_last_update_step = int(env.common_step_counter)
        env._command_curriculum_initialized = True

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # Update the curriculum from a smoothed tracking score instead of relying on a
    # single reset subset landing exactly on a global episode boundary.
    if env.common_step_counter > 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        current_tracking = torch.mean(episode_sums[env_ids]) / env.max_episode_length_s

        if env._command_curriculum_tracking_ema is None:
            env._command_curriculum_tracking_ema = current_tracking.detach()
        else:
            env._command_curriculum_tracking_ema = (
                (1.0 - ema_alpha) * env._command_curriculum_tracking_ema + ema_alpha * current_tracking.detach()
            )

        min_update_interval = max(1, int(round(env.max_episode_length * min_update_interval_episodes)))
        steps_since_update = int(env.common_step_counter - env._command_curriculum_last_update_step)

        if steps_since_update >= min_update_interval and env._command_curriculum_tracking_ema > (
            success_fraction * reward_term_cfg.weight
        ):
            delta_command = torch.tensor([-step_size, step_size], device=env.device)
            current_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
            current_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()
            if not torch.allclose(new_vel_x, current_vel_x) or not torch.allclose(new_vel_y, current_vel_y):
                env._command_curriculum_last_update_step = int(env.common_step_counter)

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def jump_reward_weight_schedule(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    root_term_name: str = "root_x_progress",
    touchdown_term_name: str = "touchdown_target_x",
    landing_term_name: str = "landing_stable",
    switch_steps: int = 20_000_000,
    early_root_weight: float = 0.5,
    late_root_weight: float = 0.1,
    early_touchdown_weight: float = 8.0,
    late_touchdown_weight: float = 10.0,
    early_landing_weight: float = 3.0,
    late_landing_weight: float = 4.0,
) -> torch.Tensor:
    """Two-stage reward schedule for jump training.

    Stage-1: stronger dense progress term for jump emergence.
    Stage-2: weaker progress and stronger touchdown/landing precision.
    """
    del env_ids  # curriculum is global; weights are shared across envs

    use_late_stage = env.common_step_counter >= int(switch_steps)

    root_weight = float(late_root_weight if use_late_stage else early_root_weight)
    touchdown_weight = float(late_touchdown_weight if use_late_stage else early_touchdown_weight)
    landing_weight = float(late_landing_weight if use_late_stage else early_landing_weight)

    root_cfg = env.reward_manager.get_term_cfg(root_term_name)
    if root_cfg.weight != root_weight:
        root_cfg.weight = root_weight
        env.reward_manager.set_term_cfg(root_term_name, root_cfg)

    touchdown_cfg = env.reward_manager.get_term_cfg(touchdown_term_name)
    if touchdown_cfg.weight != touchdown_weight:
        touchdown_cfg.weight = touchdown_weight
        env.reward_manager.set_term_cfg(touchdown_term_name, touchdown_cfg)

    landing_cfg = env.reward_manager.get_term_cfg(landing_term_name)
    if landing_cfg.weight != landing_weight:
        landing_cfg.weight = landing_weight
        env.reward_manager.set_term_cfg(landing_term_name, landing_cfg)

    return torch.tensor(root_weight, device=env.device)


def block_obstacle_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    obstacle_names: Sequence[str],
    obstacle_half_heights: Sequence[float],
    obstacle_half_lengths: Sequence[float],
    switch_steps: int = 20_000_000,
    active_x: float = 1.0,
    active_y: float = 0.0,
    hidden_x: float = 100.0,
    hidden_z: float = -10.0,
    base_clearance: float = 0.35,
    root_term_name: str | None = "root_x_progress",
    touchdown_term_name: str | None = "touchdown_target_x",
    top_pose_term_name: str | None = "block_top_pose",
    clearance_term_name: str | None = "jump_clearance",
    landing_pred_term_name: str | None = "landing_pred_xy",
    success_term_name: str | None = "jump_success",
    fail_term_name: str | None = "jump_fail",
) -> torch.Tensor:
    """Activate progressively larger obstacle blocks as training proceeds.

    The function expects multiple pre-spawned blocks with increasing size.
    Only one block level is kept active in front of the robot at a time.
    """
    del env_ids  # curriculum is global; one active block level for all envs

    num_levels = len(obstacle_names)
    if num_levels == 0:
        return torch.tensor(0.0, device=env.device)
    if num_levels != len(obstacle_half_heights):
        raise ValueError(
            f"Length mismatch: obstacle_names={num_levels}, obstacle_half_heights={len(obstacle_half_heights)}"
        )
    if num_levels != len(obstacle_half_lengths):
        raise ValueError(
            f"Length mismatch: obstacle_names={num_levels}, obstacle_half_lengths={len(obstacle_half_lengths)}"
        )

    stage = min(int(env.common_step_counter // int(switch_steps)), num_levels - 1)

    if getattr(env, "_active_block_level", None) == stage:
        return torch.tensor(float(stage), device=env.device)
    env._active_block_level = stage

    env_origins = env.scene.env_origins
    num_envs = env_origins.shape[0]
    target_x = float(active_x)
    target_z = float(obstacle_half_heights[stage] + float(base_clearance))
    top_height = float(2.0 * obstacle_half_heights[stage])
    top_half_x = float(obstacle_half_lengths[stage])

    for level, obstacle_name in enumerate(obstacle_names):
        obstacle = env.scene[obstacle_name]
        pose = obstacle.data.default_root_state[:, :7].clone()
        pose[:, 3] = 1.0
        pose[:, 4:] = 0.0

        if level == stage:
            pose[:, 0] = env_origins[:, 0] + float(active_x)
            pose[:, 1] = env_origins[:, 1] + float(active_y)
            pose[:, 2] = float(obstacle_half_heights[level])
        else:
            pose[:, 0] = env_origins[:, 0] + float(hidden_x) + 2.0 * level
            pose[:, 1] = env_origins[:, 1] + float(active_y)
            pose[:, 2] = float(hidden_z)

        obstacle.write_root_pose_to_sim(pose)

        velocity = torch.zeros((num_envs, 6), device=env.device, dtype=pose.dtype)
        obstacle.write_root_velocity_to_sim(velocity)

    # Keep reward targets aligned with active block level.
    if root_term_name is not None:
        try:
            root_cfg = env.reward_manager.get_term_cfg(root_term_name)
            if "target_x" in root_cfg.params and root_cfg.params["target_x"] != target_x:
                root_cfg.params["target_x"] = target_x
                env.reward_manager.set_term_cfg(root_term_name, root_cfg)
        except Exception:
            pass
    if touchdown_term_name is not None:
        try:
            touchdown_cfg = env.reward_manager.get_term_cfg(touchdown_term_name)
            if "target_x" in touchdown_cfg.params and touchdown_cfg.params["target_x"] != target_x:
                touchdown_cfg.params["target_x"] = target_x
                env.reward_manager.set_term_cfg(touchdown_term_name, touchdown_cfg)
        except Exception:
            pass
    if top_pose_term_name is not None:
        try:
            top_pose_cfg = env.reward_manager.get_term_cfg(top_pose_term_name)
            need_update = False
            if "target_x" in top_pose_cfg.params and top_pose_cfg.params["target_x"] != target_x:
                top_pose_cfg.params["target_x"] = target_x
                need_update = True
            if "target_z" in top_pose_cfg.params and top_pose_cfg.params["target_z"] != target_z:
                top_pose_cfg.params["target_z"] = target_z
                need_update = True
            if need_update:
                env.reward_manager.set_term_cfg(top_pose_term_name, top_pose_cfg)
        except Exception:
            pass
    if clearance_term_name is not None:
        try:
            clearance_cfg = env.reward_manager.get_term_cfg(clearance_term_name)
            if "top_height" in clearance_cfg.params and clearance_cfg.params["top_height"] != top_height:
                clearance_cfg.params["top_height"] = top_height
                env.reward_manager.set_term_cfg(clearance_term_name, clearance_cfg)
        except Exception:
            pass
    if landing_pred_term_name is not None:
        try:
            landing_pred_cfg = env.reward_manager.get_term_cfg(landing_pred_term_name)
            need_update = False
            if "target_x" in landing_pred_cfg.params and landing_pred_cfg.params["target_x"] != target_x:
                landing_pred_cfg.params["target_x"] = target_x
                need_update = True
            if "target_z" in landing_pred_cfg.params and landing_pred_cfg.params["target_z"] != top_height:
                landing_pred_cfg.params["target_z"] = top_height
                need_update = True
            if need_update:
                env.reward_manager.set_term_cfg(landing_pred_term_name, landing_pred_cfg)
        except Exception:
            pass
    if success_term_name is not None:
        try:
            success_cfg = env.reward_manager.get_term_cfg(success_term_name)
            need_update = False
            if "top_center_x" in success_cfg.params and success_cfg.params["top_center_x"] != target_x:
                success_cfg.params["top_center_x"] = target_x
                need_update = True
            if "top_height" in success_cfg.params and success_cfg.params["top_height"] != top_height:
                success_cfg.params["top_height"] = top_height
                need_update = True
            if "top_half_x" in success_cfg.params and success_cfg.params["top_half_x"] != top_half_x:
                success_cfg.params["top_half_x"] = top_half_x
                need_update = True
            if need_update:
                env.reward_manager.set_term_cfg(success_term_name, success_cfg)
        except Exception:
            pass
    if fail_term_name is not None:
        try:
            fail_cfg = env.reward_manager.get_term_cfg(fail_term_name)
            need_update = False
            if "top_center_x" in fail_cfg.params and fail_cfg.params["top_center_x"] != target_x:
                fail_cfg.params["top_center_x"] = target_x
                need_update = True
            if "top_height" in fail_cfg.params and fail_cfg.params["top_height"] != top_height:
                fail_cfg.params["top_height"] = top_height
                need_update = True
            if "top_half_x" in fail_cfg.params and fail_cfg.params["top_half_x"] != top_half_x:
                fail_cfg.params["top_half_x"] = top_half_x
                need_update = True
            if need_update:
                fail_cfg.params["support_sensor_cfg"].body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
                env.reward_manager.set_term_cfg(fail_term_name, fail_cfg)
        except Exception:
            pass

    return torch.tensor(float(stage), device=env.device)
