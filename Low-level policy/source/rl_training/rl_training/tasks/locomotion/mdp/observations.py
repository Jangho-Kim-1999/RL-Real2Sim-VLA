# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _selected_joint_ids(asset: Articulation, asset_cfg: SceneEntityCfg) -> list[int]:
    if asset_cfg.joint_ids == slice(None):
        return list(range(asset.num_joints))
    return list(asset_cfg.joint_ids)


def _resolve_hip_knee_pairs(
    env: ManagerBasedEnv,
    asset: Articulation,
    joint_ids: list[int],
) -> list[tuple[int, int]]:
    hip_token = str(getattr(env.cfg, "hip_joint_token", "HIP")).upper()
    knee_token = str(getattr(env.cfg, "knee_joint_token", "KNEE")).upper()

    names = [asset.joint_names[idx] for idx in joint_ids]
    knee_by_prefix: dict[str, int] = {}
    for local_idx, name in enumerate(names):
        upper = name.upper()
        knee_pos = upper.find(knee_token)
        if knee_pos >= 0:
            knee_by_prefix[name[:knee_pos]] = local_idx

    pairs: list[tuple[int, int]] = []
    for local_idx, name in enumerate(names):
        upper = name.upper()
        hip_pos = upper.find(hip_token)
        if hip_pos < 0:
            continue
        knee_local_idx = knee_by_prefix.get(name[:hip_pos])
        if knee_local_idx is not None:
            pairs.append((local_idx, knee_local_idx))

    pairs.sort(key=lambda pair: pair[0])
    return pairs


def _to_bispace(env: ManagerBasedEnv, values: torch.Tensor, asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    if not bool(getattr(env.cfg, "use_biarticular_hip_knee", True)):
        return values

    joint_ids = _selected_joint_ids(asset, asset_cfg)
    pairs = _resolve_hip_knee_pairs(env, asset, joint_ids)
    if not pairs:
        return values

    out = values.clone()
    for hip_idx, knee_idx in pairs:
        # qm = q1 (unchanged at hip index), qb = q1 + q2 (stored at knee index)
        out[:, knee_idx] = values[:, hip_idx] + values[:, knee_idx]
    return out


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def joint_pos_rel_bispace(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint position residual in bi-space (hip: qm=q1, knee: qb=q1+q2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return _to_bispace(env, joint_pos_rel, asset, asset_cfg)


def joint_vel_rel_bispace(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint velocity residual in bi-space (hip: dqm=dq1, knee: dqb=dq1+dq2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel_rel = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    return _to_bispace(env, joint_vel_rel, asset, asset_cfg)


def block_task_command(
    env: ManagerBasedRLEnv,
    target_x: float,
    target_z: float,
    command_name: str = "jump_mode",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return block-task command observation: [dx_to_target_x, dz_to_target_z, jump_flag]."""
    asset: Articulation = env.scene[asset_cfg.name]
    x_rel = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    z_abs = asset.data.root_pos_w[:, 2]
    dx = float(target_x) - x_rel
    dz = float(target_z) - z_abs

    jump_flag = torch.ones_like(dx)
    if command_name:
        try:
            cmd = env.command_manager.get_command(command_name)
            if cmd.ndim == 1:
                cmd = cmd.unsqueeze(-1)
            jump_flag = cmd[:, 0].to(dtype=dx.dtype)
        except KeyError:
            pass

    return torch.stack((dx, dz, jump_flag), dim=1)


def _cached_obs_body_ids(
    env: ManagerBasedEnv,
    asset: Articulation,
    cache_key: str,
    body_names: tuple[str, ...] | list[str],
) -> torch.Tensor:
    """Resolve body names once and reuse ids across observation calls."""
    if not hasattr(env, "_obs_body_id_cache") or env._obs_body_id_cache is None:
        env._obs_body_id_cache = {}
    key = (cache_key, tuple(body_names))
    if key not in env._obs_body_id_cache:
        body_ids, _ = asset.find_bodies(tuple(body_names), preserve_order=True)
        env._obs_body_id_cache[key] = torch.as_tensor(body_ids, device=asset.device, dtype=torch.long)
    return env._obs_body_id_cache[key]


def fr_push_task_state(
    env: ManagerBasedRLEnv,
    target_x: float,
    target_y: float,
    start_x: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("push_object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fr_foot_body_name: str = "FR_foot",
) -> torch.Tensor:
    """Task observation for FR-foot object pushing.

    Returns:
        [object_from_base_b(3), object_target_error_xy(2), object_xy_velocity_w(2),
        fr_foot_to_object_b(3), object_progress(1)].
    """
    robot: Articulation = env.scene[asset_cfg.name]
    push_object: RigidObject = env.scene[object_cfg.name]

    foot_ids = _cached_obs_body_ids(env, robot, "_fr_push_foot", (fr_foot_body_name,))
    fr_foot_pos_w = robot.data.body_link_pos_w[:, foot_ids[0], :]

    object_pos_w = push_object.data.root_pos_w[:, :3]
    base_pos_w = robot.data.root_pos_w[:, :3]
    yaw_quat = math_utils.yaw_quat(robot.data.root_quat_w)

    object_from_base_b = math_utils.quat_apply_inverse(yaw_quat, object_pos_w - base_pos_w)
    fr_foot_to_object_b = math_utils.quat_apply_inverse(yaw_quat, object_pos_w - fr_foot_pos_w)

    object_pos_rel = object_pos_w - env.scene.env_origins
    target_error_xy = torch.stack(
        (
            torch.full_like(object_pos_rel[:, 0], float(target_x)) - object_pos_rel[:, 0],
            torch.full_like(object_pos_rel[:, 1], float(target_y)) - object_pos_rel[:, 1],
        ),
        dim=1,
    )
    object_vel_xy_w = push_object.data.root_lin_vel_w[:, :2]
    progress_denom = max(float(target_x) - float(start_x), 1.0e-6)
    progress = torch.clamp((object_pos_rel[:, 0] - float(start_x)) / progress_denom, 0.0, 1.0).unsqueeze(1)

    return torch.cat(
        (
            object_from_base_b,
            target_error_xy,
            object_vel_xy_w,
            fr_foot_to_object_b,
            progress,
        ),
        dim=1,
    )


def _obs_contact_sensor_total_forces_w(contact_sensor: ContactSensor, body_ids: torch.Tensor) -> torch.Tensor:
    """Return current total contact force vectors for selected contact-sensor bodies."""
    if getattr(contact_sensor.data, "force_matrix_w", None) is not None:
        normal = torch.nan_to_num(contact_sensor.data.force_matrix_w[:, body_ids], nan=0.0).sum(dim=2)
        friction_data = getattr(contact_sensor.data, "friction_forces_w", None)
        if friction_data is not None:
            friction = torch.nan_to_num(friction_data[:, body_ids], nan=0.0).sum(dim=2)
        else:
            friction = torch.zeros_like(normal)
        return normal + friction
    if getattr(contact_sensor.data, "net_forces_w", None) is not None:
        return torch.nan_to_num(contact_sensor.data.net_forces_w[:, body_ids], nan=0.0)
    raise RuntimeError("Contact sensor does not expose current contact force vectors.")


def fr_wall_push_task_state(
    env: ManagerBasedRLEnv,
    wall_contact_x: float,
    target_y: float,
    target_z: float,
    target_force: float,
    force_axis: int = 0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wall_contact_forces", body_names=["FR_foot"]),
    fr_foot_body_name: str = "FR_foot",
) -> torch.Tensor:
    """Task observation for pushing a fixed wall with FR foot.

    Returns:
        [wall_target_from_base_b(3), fr_foot_to_wall_target_b(3), normalized_wall_force(1)].
    """
    robot: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_ids = _cached_obs_body_ids(env, robot, "_fr_wall_push_foot", (fr_foot_body_name,))
    fr_foot_pos_w = robot.data.body_link_pos_w[:, foot_ids[0], :]
    wall_target_w = env.scene.env_origins + torch.tensor(
        [float(wall_contact_x), float(target_y), float(target_z)], device=env.device, dtype=fr_foot_pos_w.dtype
    )

    base_pos_w = robot.data.root_pos_w[:, :3]
    yaw_quat = math_utils.yaw_quat(robot.data.root_quat_w)
    wall_target_from_base_b = math_utils.quat_apply_inverse(yaw_quat, wall_target_w - base_pos_w)
    fr_foot_to_wall_target_b = math_utils.quat_apply_inverse(yaw_quat, wall_target_w - fr_foot_pos_w)

    body_ids = sensor_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = torch.arange(contact_sensor.data.net_forces_w.shape[1], device=env.device)
    else:
        body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
    total_forces_w = _obs_contact_sensor_total_forces_w(contact_sensor, body_ids)
    normal_force = torch.abs(total_forces_w[:, :, int(force_axis)].sum(dim=1))
    normalized_force = (normal_force / max(float(target_force), 1.0e-6)).unsqueeze(1)

    return torch.cat((wall_target_from_base_b, fr_foot_to_wall_target_b, normalized_force), dim=1)


def fr_wall_push_actor_task_state(
    env: ManagerBasedRLEnv,
    wall_contact_x: float,
    command_force: float,
    force_axis: int = 0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_force_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FR_foot"]),
    fr_foot_body_name: str = "FR_foot",
    wall_cfg: SceneEntityCfg | None = None,
    wall_half_x: float = 0.0,
) -> torch.Tensor:
    """Actor task observation for FR fixed-wall force control.

    Returns 8 dims:
        [wall_x_from_base_b(1), fr_foot_pos_from_base_b(3), command_force(1), fr_foot_total_force_w(3)].
    """
    robot: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[foot_force_sensor_cfg.name]

    foot_ids = _cached_obs_body_ids(env, robot, "_fr_wall_actor_foot", (fr_foot_body_name,))
    fr_foot_pos_w = robot.data.body_link_pos_w[:, foot_ids[0], :]
    base_pos_w = robot.data.root_pos_w[:, :3]
    yaw_quat = math_utils.yaw_quat(robot.data.root_quat_w)

    wall_point_w = base_pos_w.clone()
    if wall_cfg is not None:
        wall: RigidObject = env.scene[wall_cfg.name]
        wall_point_w[:, 0] = wall.data.root_pos_w[:, 0] - float(wall_half_x)
    else:
        wall_point_w[:, 0] = env.scene.env_origins[:, 0] + float(wall_contact_x)
    wall_from_base_b = math_utils.quat_apply_inverse(yaw_quat, wall_point_w - base_pos_w)
    fr_foot_from_base_b = math_utils.quat_apply_inverse(yaw_quat, fr_foot_pos_w - base_pos_w)

    body_ids = foot_force_sensor_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = torch.arange(contact_sensor.data.net_forces_w.shape[1], device=env.device)
    else:
        body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
    fr_foot_force_w = _obs_contact_sensor_total_forces_w(contact_sensor, body_ids).sum(dim=1)
    command_force_tensor = torch.full(
        (env.num_envs, 1), float(command_force), device=env.device, dtype=fr_foot_pos_w.dtype
    )

    return torch.cat(
        (
            wall_from_base_b[:, 0:1],
            fr_foot_from_base_b,
            command_force_tensor,
            fr_foot_force_w,
        ),
        dim=1,
    )


def fr_wall_push_critic_task_state(
    env: ManagerBasedRLEnv,
    wall_contact_x: float,
    command_force: float,
    force_axis: int = 0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_force_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FR_foot"]),
    wall_force_sensor_cfg: SceneEntityCfg = SceneEntityCfg("wall_contact_forces", body_names=["FR_foot"]),
    fr_foot_body_name: str = "FR_foot",
    wall_cfg: SceneEntityCfg | None = None,
    wall_half_x: float = 0.0,
) -> torch.Tensor:
    """Critic task observation: actor task state plus wall-only normal force."""
    actor_obs = fr_wall_push_actor_task_state(
        env=env,
        wall_contact_x=wall_contact_x,
        command_force=command_force,
        force_axis=force_axis,
        asset_cfg=asset_cfg,
        foot_force_sensor_cfg=foot_force_sensor_cfg,
        fr_foot_body_name=fr_foot_body_name,
        wall_cfg=wall_cfg,
        wall_half_x=wall_half_x,
    )

    wall_contact_sensor: ContactSensor = env.scene.sensors[wall_force_sensor_cfg.name]
    body_ids = wall_force_sensor_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = torch.arange(wall_contact_sensor.data.net_forces_w.shape[1], device=env.device)
    else:
        body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
    wall_forces_w = _obs_contact_sensor_total_forces_w(wall_contact_sensor, body_ids)
    wall_normal_force = torch.abs(wall_forces_w[:, :, int(force_axis)].sum(dim=1)).unsqueeze(1)
    return torch.cat((actor_obs, wall_normal_force), dim=1)


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def prev_action(env: ManagerBasedEnv) -> torch.Tensor:
    """Previous action tensor from the action manager."""
    return env.action_manager.prev_action

def base_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height in world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]
