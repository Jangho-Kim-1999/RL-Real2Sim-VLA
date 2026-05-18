# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_root_state_uniform_with_foot_projection(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    min_foot_clearance: float = 0.01,
    foot_body_names: str | Sequence[str] = (".*_foot", ".*FOOT"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset root state and lift base z so the lowest foot is above the terrain origin."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    root_states = asset.data.default_root_state[env_ids].clone()
    pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_ranges = torch.tensor(pose_range_list, device=asset.device)
    rand_pose = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=asset.device
    )

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_pose[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_pose[:, 3], rand_pose[:, 4], rand_pose[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    positions = _lift_positions_to_clear_feet(
        env, asset, env_ids, positions, orientations, foot_body_names, min_foot_clearance
    )

    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=asset.device)
    rand_vel = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_vel

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def lift_root_state_to_clear_feet(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    min_foot_clearance: float = 0.01,
    foot_body_names: str | Sequence[str] = (".*_foot", ".*FOOT"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Lift the current root pose after joint reset if any foot is below the terrain origin."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    positions = asset.data.root_pos_w[env_ids].clone()
    orientations = asset.data.root_quat_w[env_ids].clone()
    positions = _lift_positions_to_clear_feet(
        env, asset, env_ids, positions, orientations, foot_body_names, min_foot_clearance
    )
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)


def _lift_positions_to_clear_feet(
    env: ManagerBasedEnv,
    asset: RigidObject | Articulation,
    env_ids: torch.Tensor,
    positions: torch.Tensor,
    orientations: torch.Tensor,
    foot_body_names: str | Sequence[str],
    min_foot_clearance: float,
) -> torch.Tensor:
    foot_body_ids, _ = asset.find_bodies(foot_body_names, preserve_order=True)
    if len(foot_body_ids) == 0:
        return positions

    root_pos_w = asset.data.root_link_pos_w[env_ids]
    root_quat_w = asset.data.root_link_quat_w[env_ids]
    feet_pos_w = asset.data.body_link_pos_w[env_ids][:, foot_body_ids, :]

    feet_offset_w = feet_pos_w - root_pos_w.unsqueeze(1)
    feet_offset_b = math_utils.quat_apply_inverse(
        root_quat_w[:, None, :].expand(-1, len(foot_body_ids), -1).reshape(-1, 4),
        feet_offset_w.reshape(-1, 3),
    ).reshape(len(env_ids), len(foot_body_ids), 3)

    feet_pred_w = math_utils.quat_apply(
        orientations[:, None, :].expand(-1, len(foot_body_ids), -1).reshape(-1, 4),
        feet_offset_b.reshape(-1, 3),
    ).reshape(len(env_ids), len(foot_body_ids), 3)
    feet_pred_w += positions.unsqueeze(1)

    min_feet_z = feet_pred_w[..., 2].amin(dim=1)
    ground_z = env.scene.env_origins[env_ids, 2]
    z_lift = (ground_z + float(min_foot_clearance) - min_feet_z).clamp_min(0.0)

    lifted_positions = positions.clone()
    lifted_positions[:, 2] += z_lift
    return lifted_positions


def reset_asset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
    them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_object_state[env_ids].clone()

    root_states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    asset.write_object_state_to_sim(root_states, env_ids=env_ids)


def reset_robot_with_cones(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, list],  # x, y, z, yaw
    asset_robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cone_red_cfg: SceneEntityCfg = SceneEntityCfg("cone_red"),
    asset_cone_green_cfg: SceneEntityCfg = SceneEntityCfg("cone_green"),
    asset_cone_blue_cfg: SceneEntityCfg = SceneEntityCfg("cone_blue"),
):
    asset_robot: RigidObject | Articulation = env.scene[asset_robot_cfg.name]
    asset_cone_red: RigidObject | Articulation = env.scene[asset_cone_red_cfg.name]
    asset_cone_green: RigidObject | Articulation = env.scene[asset_cone_green_cfg.name]
    asset_cone_blue: RigidObject | Articulation = env.scene[asset_cone_blue_cfg.name]
    # get default root state
    default_root_states = asset_robot.data.default_root_state[env_ids].clone()

    x_range_list = pose_range.get("x")
    y_range_list = pose_range.get("y")
    z_range_list = pose_range.get("z")
    yaw_range_list = pose_range.get("yaw")
    assert (
        len(x_range_list) == len(y_range_list) == len(z_range_list) == len(yaw_range_list)
    ), "The length of x, y, z, and yaw ranges should be the same."

    def tighten_range(range_list, tightness=0.2):
        tightened_ranges = []
        for start, end in range_list:
            new_start = start + tightness
            new_end = end - tightness
            if new_start < new_end:
                tightened_ranges.append((new_start, new_end))
            else:
                assert False, "Invalid range"
        return tightened_ranges

    x_range_list_robot = tighten_range(x_range_list)
    y_range_list_robot = tighten_range(y_range_list)
    z_range_list_robot = z_range_list
    yaw_range_list_robot = yaw_range_list

    random_robot_list = []
    random_cone_red_list = []
    random_cone_green_list = []
    random_cone_blue_list = []

    for i in range(len(env_ids)):
        chosen_indices = torch.randperm(len(x_range_list))[:4]
        for j in range(4):
            idx = chosen_indices[j]

            if j == 0:
                x_range = x_range_list_robot[idx]
                y_range = y_range_list_robot[idx]
                z_range = z_range_list_robot[idx]
                yaw_range = yaw_range_list_robot[idx]
            else:
                x_range = x_range_list[idx]
                y_range = y_range_list[idx]
                z_range = z_range_list[idx]
                yaw_range = yaw_range_list[idx]

            if j == 0:
                ranges = torch.tensor([x_range, y_range, z_range, yaw_range], device=asset_robot.device)
                rand_samples_robot = math_utils.sample_uniform(
                    ranges[:, 0], ranges[:, 1], (4), device=asset_robot.device
                )
                random_robot_list.append(rand_samples_robot)
            elif j == 1:
                ranges = torch.tensor([x_range, y_range, z_range], device=asset_robot.device)
                rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (3), device=asset_robot.device)
                random_cone_red_list.append(rand_samples)
            elif j == 2:
                ranges = torch.tensor([x_range, y_range, z_range], device=asset_robot.device)
                rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (3), device=asset_robot.device)
                random_cone_green_list.append(rand_samples)
            elif j == 3:
                ranges = torch.tensor([x_range, y_range, z_range], device=asset_robot.device)
                rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (3), device=asset_robot.device)
                random_cone_blue_list.append(rand_samples)

    robot_samples = torch.stack(random_robot_list, dim=0)  # (len(env_ids), 4)
    cone_red_samples = torch.stack(random_cone_red_list, dim=0)  # (len(env_ids), 3)
    cone_green_samples = torch.stack(random_cone_green_list, dim=0)  # (len(env_ids), 3)
    cone_blue_samples = torch.stack(random_cone_blue_list, dim=0)  # (len(env_ids), 3)

    rand_samples = torch.zeros((len(env_ids), 6), device=asset_robot.device)
    rand_samples[:, 0:3] = robot_samples[:, 0:3]
    rand_samples[:, 5] = robot_samples[:, 3]

    positions = default_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(default_root_states[:, 3:7], orientations_delta)
    velocities = default_root_states[:, 7:13]

    root_states = asset_cone_red.data.default_object_state[env_ids].clone()
    root_states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    root_states[..., :3] += cone_red_samples.unsqueeze(1)
    asset_cone_red.write_object_state_to_sim(root_states, env_ids=env_ids)

    root_states = asset_cone_green.data.default_object_state[env_ids].clone()
    root_states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    root_states[..., :3] += cone_green_samples.unsqueeze(1)
    asset_cone_green.write_object_state_to_sim(root_states, env_ids=env_ids)

    root_states = asset_cone_blue.data.default_object_state[env_ids].clone()
    root_states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    root_states[..., :3] += cone_blue_samples.unsqueeze(1)
    asset_cone_blue.write_object_state_to_sim(root_states, env_ids=env_ids)


def reset_cones(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, list],  # x, y, z, yaw
    asset_robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cone_red_cfg: SceneEntityCfg = SceneEntityCfg("cone_red"),
    asset_cone_green_cfg: SceneEntityCfg = SceneEntityCfg("cone_green"),
    asset_cone_blue_cfg: SceneEntityCfg = SceneEntityCfg("cone_blue"),
):
    asset_robot: RigidObject | Articulation = env.scene[asset_robot_cfg.name]
    asset_cone_red: RigidObject | Articulation = env.scene[asset_cone_red_cfg.name]
    asset_cone_green: RigidObject | Articulation = env.scene[asset_cone_green_cfg.name]
    asset_cone_blue: RigidObject | Articulation = env.scene[asset_cone_blue_cfg.name]

    x_range_list = pose_range.get("x")
    y_range_list = pose_range.get("y")
    z_range_list = pose_range.get("z")
    yaw_range_list = pose_range.get("yaw")
    assert (
        len(x_range_list) == len(y_range_list) == len(z_range_list) == len(yaw_range_list)
    ), "The length of x, y, z, and yaw ranges should be the same."

    random_robot_list = []
    random_cone_red_list = []
    random_cone_green_list = []
    random_cone_blue_list = []

    for i in range(len(env_ids)):
        chosen_indices = torch.randperm(len(x_range_list))[:3]
        for j in range(3):
            idx = chosen_indices[j]

            x_range = x_range_list[idx]
            y_range = y_range_list[idx]
            z_range = z_range_list[idx]
            yaw_range = yaw_range_list[idx]

            if j == 0:
                ranges = torch.tensor([x_range, y_range, z_range], device=asset_robot.device)
                rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (3), device=asset_robot.device)
                random_cone_red_list.append(rand_samples)
            elif j == 1:
                ranges = torch.tensor([x_range, y_range, z_range], device=asset_robot.device)
                rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (3), device=asset_robot.device)
                random_cone_green_list.append(rand_samples)
            elif j == 2:
                ranges = torch.tensor([x_range, y_range, z_range], device=asset_robot.device)
                rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (3), device=asset_robot.device)
                random_cone_blue_list.append(rand_samples)

    cone_red_samples = torch.stack(random_cone_red_list, dim=0)  # (len(env_ids), 3)
    cone_green_samples = torch.stack(random_cone_green_list, dim=0)  # (len(env_ids), 3)
    cone_blue_samples = torch.stack(random_cone_blue_list, dim=0)  # (len(env_ids), 3)

    root_states = asset_cone_red.data.default_object_state[env_ids].clone()
    root_states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    root_states[..., :3] += cone_red_samples.unsqueeze(1)
    asset_cone_red.write_object_state_to_sim(root_states, env_ids=env_ids)

    root_states = asset_cone_green.data.default_object_state[env_ids].clone()
    root_states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    root_states[..., :3] += cone_green_samples.unsqueeze(1)
    asset_cone_green.write_object_state_to_sim(root_states, env_ids=env_ids)

    root_states = asset_cone_blue.data.default_object_state[env_ids].clone()
    root_states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    root_states[..., :3] += cone_blue_samples.unsqueeze(1)
    asset_cone_blue.write_object_state_to_sim(root_states, env_ids=env_ids)


def reset_single_cone(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cone_red"),
):
    """Randomize one cone target per environment."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_object_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_pos = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    root_states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    root_states[..., :3] += rand_pos.unsqueeze(1)
    asset.write_object_state_to_sim(root_states, env_ids=env_ids)


def reset_root_state_uniform_custom(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, list],
    pose_range_prob: list,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    x_range_list = pose_range.get("x")
    y_range_list = pose_range.get("y")
    yaw_range_list = pose_range.get("yaw")
    z_range_list = pose_range.get("z")
    assert (
        len(x_range_list) == len(y_range_list) == len(yaw_range_list) == len(pose_range_prob)
    ), "The length of x, y, and yaw ranges should be the same."

    rand_idx = torch.multinomial(torch.tensor(pose_range_prob, device=asset.device), len(env_ids), replacement=True)
    rand_samples_list = []

    for i in range(len(x_range_list)):
        x_range = x_range_list[i]
        y_range = y_range_list[i]
        z_range = z_range_list[i]
        yaw_range = yaw_range_list[i]

        ranges = torch.tensor([x_range, y_range, z_range, yaw_range], device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 4), device=asset.device)
        rand_samples_list.append(rand_samples)

    multi_samples = torch.stack(rand_samples_list, dim=0)  # (3, len(env_ids), 3)
    final_samples = []

    for i, env_id in enumerate(env_ids):
        sampled_positions = multi_samples[rand_idx[i], i, :]
        final_samples.append(sampled_positions)

    final_samples = torch.stack(final_samples, dim=0).to(asset.device)
    rand_samples = torch.zeros((len(env_ids), 6), device=asset.device)
    rand_samples[:, 0:3] = final_samples[:, 0:3]
    rand_samples[:, 5] = final_samples[:, 3]

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
