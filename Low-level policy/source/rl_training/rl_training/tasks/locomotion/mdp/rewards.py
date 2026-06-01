# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _use_biarticular_hip_knee(env: ManagerBasedRLEnv) -> bool:
    return bool(getattr(env.cfg, "use_biarticular_hip_knee", True))


def _selected_joint_ids(asset: Articulation, asset_cfg: SceneEntityCfg) -> list[int]:
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice) and joint_ids == slice(None):
        return list(range(asset.num_joints))
    if torch.is_tensor(joint_ids):
        return [int(i) for i in joint_ids.tolist()]
    return [int(i) for i in joint_ids]


def _resolve_hip_knee_pairs(
    env: ManagerBasedRLEnv,
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


def _bi_pairs_for_cfg(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
) -> list[tuple[int, int]]:
    if not _use_biarticular_hip_knee(env):
        return []
    joint_ids = _selected_joint_ids(asset, asset_cfg)
    return _resolve_hip_knee_pairs(env, asset, joint_ids)


def _apply_bispace_state(values: torch.Tensor, pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return values
    out = values.clone()
    for hip_idx, knee_idx in pairs:
        # qm = q_hip (unchanged), qb = q_hip + q_knee (stored at knee index)
        out[:, knee_idx] = values[:, hip_idx] + values[:, knee_idx]
    return out


def _apply_bispace_torque(values: torch.Tensor, pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return values
    out = values.clone()
    for hip_idx, knee_idx in pairs:
        # tau_m = tau_hip - tau_knee, tau_b = tau_knee
        out[:, hip_idx] = values[:, hip_idx] - values[:, knee_idx]
        out[:, knee_idx] = values[:, knee_idx]
    return out


def _apply_bispace_pos_limits(
    lower: torch.Tensor,
    upper: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not pairs:
        return lower, upper
    lower_out = lower.clone()
    upper_out = upper.clone()
    for hip_idx, knee_idx in pairs:
        lower_out[:, knee_idx] = lower[:, hip_idx] + lower[:, knee_idx]
        upper_out[:, knee_idx] = upper[:, hip_idx] + upper[:, knee_idx]
    return lower_out, upper_out


def _apply_bispace_abs_limits(limits: torch.Tensor, pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return limits
    out = limits.clone()
    for hip_idx, knee_idx in pairs:
        out[:, knee_idx] = limits[:, hip_idx] + limits[:, knee_idx]
    return out


def _command_phase_mask(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    active_when_jump: bool = True,
) -> torch.Tensor:
    """Return reward mask from command flag (1: jump, 0: stand)."""
    if command_name is None:
        return torch.ones(env.num_envs, device=env.device)
    try:
        cmd = env.command_manager.get_command(command_name)
    except KeyError:
        return torch.ones(env.num_envs, device=env.device)
    if cmd.ndim == 1:
        cmd = cmd.unsqueeze(-1)
    jump = cmd[:, 0] > 0.5
    return jump.float() if active_when_jump else (~jump).float()


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_axis_ref_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    command_index: int,
    vel_axis: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of one base linear-velocity axis command using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, int(command_index)]
    vel = asset.data.root_lin_vel_b[:, int(vel_axis)]
    err = torch.square(cmd - vel)
    return torch.exp(-err / std**2)


def track_stance_x_zero_exp(
    env: ManagerBasedRLEnv,
    std: float,
    foot_body_names: tuple[str, str, str, str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking root x=0 relative to average support point in yaw frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_ids = _cached_body_ids(env, asset, "_stance_track_feet", foot_body_names)
    foot_center_w = asset.data.body_link_pos_w[:, foot_ids, :].mean(dim=1)
    delta_w = asset.data.root_pos_w - foot_center_w
    delta_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), delta_w)
    x_err = torch.square(delta_b[:, 0])
    return torch.exp(-x_err / std**2)


def _body_pose_pos_ref_component(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_index: int,
) -> torch.Tensor:
    """Return one position reference component from a command term that exposes pos_ref_buffer."""
    term = env.command_manager.get_term(command_name)
    if hasattr(term, "pos_ref_buffer") and torch.is_tensor(term.pos_ref_buffer):
        return term.pos_ref_buffer[:, int(command_index)]
    # Fallback keeps reward numerically safe if the command term does not expose pos_ref_buffer.
    return torch.zeros(env.num_envs, device=env.device)


def track_stance_z_pos_ref_from_command_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    foot_body_names: tuple[str, str, str, str],
    command_index: int = 1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking root z against position reference generated by the command term."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_ids = _cached_body_ids(env, asset, "_stance_track_feet", foot_body_names)
    foot_center_w = asset.data.body_link_pos_w[:, foot_ids, :].mean(dim=1)
    delta_w = asset.data.root_pos_w - foot_center_w
    delta_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), delta_w)
    z_ref = _body_pose_pos_ref_component(env, command_name, command_index)
    z_err = torch.square(delta_b[:, 2] - z_ref)
    return torch.exp(-z_err / std**2)


def track_base_pitch_pos_ref_from_command_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    command_index: int = 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking base pitch against position reference generated by the command term."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    pitch_ref = _body_pose_pos_ref_component(env, command_name, command_index)
    pitch_err = torch.square(pitch - pitch_ref)
    return torch.exp(-pitch_err / std**2)


def orientation_control_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    roll_index: int = 0,
    pitch_index: int = 1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward attitude tracking from roll/pitch command using projected-gravity error."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    roll_ref = cmd[:, int(roll_index)]
    pitch_ref = cmd[:, int(pitch_index)]
    yaw_ref = torch.zeros_like(roll_ref)
    desired_base_quat = math_utils.quat_from_euler_xyz(roll_ref, pitch_ref, yaw_ref)
    gravity_world = torch.zeros(env.num_envs, 3, device=env.device, dtype=cmd.dtype)
    gravity_world[:, 2] = -1.0
    desired_projected_gravity = math_utils.quat_apply_inverse(desired_base_quat, gravity_world)
    gravity_xy_err = torch.sum(
        torch.square(asset.data.projected_gravity_b[:, :2] - desired_projected_gravity[:, :2]),
        dim=1,
    )
    return torch.exp(-gravity_xy_err / std**2)


def base_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward base-height tracking with exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    err = torch.square(base_height - float(target_height))
    return torch.exp(-err / std**2)


def track_ang_vel_axis_ref_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    command_index: int,
    ang_axis: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of one base angular-velocity axis command using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, int(command_index)]
    ang_vel = asset.data.root_ang_vel_b[:, int(ang_axis)]
    err = torch.square(cmd - ang_vel)
    return torch.exp(-err / std**2)


def _cached_sensor_body_ids(
    env: ManagerBasedRLEnv,
    sensor: ContactSensor,
    cache_key: str,
    body_names: tuple[str, ...] | list[str],
) -> torch.Tensor:
    """Resolve sensor body ids once and reuse them across calls."""
    if not hasattr(env, "_reward_sensor_body_id_cache") or env._reward_sensor_body_id_cache is None:
        env._reward_sensor_body_id_cache = {}
    key = (cache_key, tuple(body_names))
    if key not in env._reward_sensor_body_id_cache:
        body_ids, _ = sensor.find_bodies(tuple(body_names), preserve_order=True)
        env._reward_sensor_body_id_cache[key] = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
    return env._reward_sensor_body_id_cache[key]


def _contact_sensor_total_forces_w(contact_sensor: ContactSensor, body_ids: torch.Tensor) -> torch.Tensor:
    """Return current total contact force vectors for selected bodies."""
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


def diagonal_grf_balance(
    env: ManagerBasedRLEnv,
    paired_body_names: tuple[tuple[str, str], ...],
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ground_reaction_forces"),
    command_name: str | None = None,
    command_threshold: float = 0.1,
    pair_load_threshold: float = 10.0,
    eps: float = 1.0,
) -> torch.Tensor:
    """Penalize diagonal-pair GRF imbalance.

    For each configured pair, compute a normalized vertical-load asymmetry:
    ``|Fz_a - Fz_b| / (Fz_a + Fz_b + eps)``.
    Pairs with negligible total load are ignored.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    flat_body_names = tuple(name for pair in paired_body_names for name in pair)
    body_ids = _cached_sensor_body_ids(env, contact_sensor, f"diagonal_grf_balance:{sensor_cfg.name}", flat_body_names)
    if body_ids.numel() != len(flat_body_names):
        raise RuntimeError(
            f"Failed to resolve all GRF pair bodies for {paired_body_names}. Resolved {body_ids.numel()} bodies."
        )

    total_forces_w = _contact_sensor_total_forces_w(contact_sensor, body_ids)
    vertical_force = total_forces_w[..., 2].clamp_min(0.0)

    reward = torch.zeros(env.num_envs, device=vertical_force.device, dtype=vertical_force.dtype)
    for pair_idx in range(len(paired_body_names)):
        force_a = vertical_force[:, 2 * pair_idx]
        force_b = vertical_force[:, 2 * pair_idx + 1]
        pair_load = force_a + force_b
        pair_penalty = torch.abs(force_a - force_b) / torch.clamp(pair_load, min=float(eps))
        if pair_load_threshold > 0.0:
            pair_penalty = torch.where(
                pair_load > float(pair_load_threshold), pair_penalty, torch.zeros_like(pair_penalty)
            )
        reward += pair_penalty

    if command_name is not None:
        reward *= (torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > float(command_threshold)).float()

    return reward


def inter_diagonal_load_balance(
    env: ManagerBasedRLEnv,
    paired_body_names: tuple[tuple[str, str], ...],
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ground_reaction_forces"),
    command_name: str | None = None,
    command_threshold: float = 0.1,
    total_load_threshold: float = 10.0,
    eps: float = 1.0,
) -> torch.Tensor:
    """Penalize total vertical-load imbalance between two diagonal pairs.

    This term compares the summed vertical GRF of the two configured diagonal pairs:
    ``|(Fz_a0 + Fz_a1) - (Fz_b0 + Fz_b1)| / (Fz_all + eps)``.
    """
    if len(paired_body_names) != 2:
        raise RuntimeError(
            f"inter_diagonal_load_balance expects exactly two diagonal pairs, got {len(paired_body_names)}."
        )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    flat_body_names = tuple(name for pair in paired_body_names for name in pair)
    body_ids = _cached_sensor_body_ids(
        env, contact_sensor, f"inter_diagonal_load_balance:{sensor_cfg.name}", flat_body_names
    )
    if body_ids.numel() != len(flat_body_names):
        raise RuntimeError(
            f"Failed to resolve all GRF pair bodies for {paired_body_names}. Resolved {body_ids.numel()} bodies."
        )

    total_forces_w = _contact_sensor_total_forces_w(contact_sensor, body_ids)
    vertical_force = total_forces_w[..., 2].clamp_min(0.0)

    pair_load_0 = vertical_force[:, 0] + vertical_force[:, 1]
    pair_load_1 = vertical_force[:, 2] + vertical_force[:, 3]
    total_load = pair_load_0 + pair_load_1
    reward = torch.abs(pair_load_0 - pair_load_1) / torch.clamp(total_load, min=float(eps))

    if total_load_threshold > 0.0:
        reward = torch.where(total_load > float(total_load_threshold), reward, torch.zeros_like(reward))

    if command_name is not None:
        reward *= (
            torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > float(command_threshold)
        ).float()

    return reward


def track_stance_x_ref_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    foot_body_names: tuple[str, str, str, str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking root x relative to the average support point in the yaw frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_ids = _cached_body_ids(env, asset, "_stance_track_feet", foot_body_names)
    foot_center_w = asset.data.body_link_pos_w[:, foot_ids, :].mean(dim=1)
    delta_w = asset.data.root_pos_w - foot_center_w
    delta_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), delta_w)
    x_ref = env.command_manager.get_command(command_name)[:, 0]
    x_err = torch.square(delta_b[:, 0] - x_ref)
    return torch.exp(-x_err / std**2)


def track_stance_z_ref_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    foot_body_names: tuple[str, str, str, str],
    command_index: int = 1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking root z relative to the average support point in the yaw frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_ids = _cached_body_ids(env, asset, "_stance_track_feet", foot_body_names)
    foot_center_w = asset.data.body_link_pos_w[:, foot_ids, :].mean(dim=1)
    delta_w = asset.data.root_pos_w - foot_center_w
    delta_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), delta_w)
    z_ref = env.command_manager.get_command(command_name)[:, int(command_index)]
    z_err = torch.square(delta_b[:, 2] - z_ref)
    return torch.exp(-z_err / std**2)


def track_base_pitch_ref_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    command_index: int = 1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking desired base pitch."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    pitch_ref = env.command_manager.get_command(command_name)[:, int(command_index)]
    pitch_err = torch.square(pitch - pitch_ref)
    return torch.exp(-pitch_err / std**2)


def track_base_yaw_ref_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    command_index: int = 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking desired base yaw."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw_ref = env.command_manager.get_command(command_name)[:, int(command_index)]
    yaw_err = torch.square(yaw - yaw_ref)
    return torch.exp(-yaw_err / std**2)


def base_roll_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-zero base roll."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, _, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.square(roll)


def base_yaw_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-zero base yaw."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.square(yaw)


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint power in bi-space for hip/knee (serial elsewhere)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
    joint_vel = _apply_bispace_state(joint_vel, pairs)
    applied_torque = _apply_bispace_torque(applied_torque, pairs)
    reward = torch.sum(
        torch.abs(joint_vel * applied_torque),
        dim=1,
    )
    return reward


def joint_power_no_comp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint power excluding tau_comp.

    Joint velocity is evaluated in bi-space for HIP/KNEE and serial elsewhere.
    Torque uses ``tau_no_comp`` when provided by the runtime env.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
    joint_vel = _apply_bispace_state(joint_vel, pairs)
    tau_no_comp = _joint_torque_no_comp_for_cfg(env, asset, asset_cfg)
    return torch.sum(torch.abs(joint_vel * tau_no_comp), dim=1)


def _joint_torque_no_comp_for_cfg(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return tau_no_comp for the selected joints.

    If the runtime env does not expose ``_last_tau_no_comp``, fall back to the
    applied torque path used by the default reward implementation.
    """
    tau_no_comp = getattr(env, "_last_tau_no_comp", None)
    if tau_no_comp is None:
        applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
        pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
        return _apply_bispace_torque(applied_torque, pairs)

    selected_joint_ids = _selected_joint_ids(asset, asset_cfg)
    env_joint_ids = getattr(env, "_joint_ids", None)
    if env_joint_ids is None:
        if tau_no_comp.shape[1] == asset.num_joints:
            return tau_no_comp[:, selected_joint_ids]
        applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
        pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
        return _apply_bispace_torque(applied_torque, pairs)

    if torch.is_tensor(env_joint_ids):
        env_joint_ids_list = [int(joint_id) for joint_id in env_joint_ids.tolist()]
    else:
        env_joint_ids_list = [int(joint_id) for joint_id in env_joint_ids]

    id_to_local = {joint_id: local_idx for local_idx, joint_id in enumerate(env_joint_ids_list)}
    local_joint_ids = [id_to_local[joint_id] for joint_id in selected_joint_ids if joint_id in id_to_local]
    if len(local_joint_ids) != len(selected_joint_ids):
        applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
        pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
        return _apply_bispace_torque(applied_torque, pairs)

    return tau_no_comp[:, local_joint_ids]


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize squared applied torque in bi-space for hip/knee (serial elsewhere)."""
    asset: Articulation = env.scene[asset_cfg.name]
    applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
    applied_torque = _apply_bispace_torque(applied_torque, pairs)
    return torch.sum(torch.square(applied_torque), dim=1)


def joint_torques_no_comp_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared torque excluding tau_comp.

    In the custom runtime env this uses ``env._last_tau_no_comp``, which already
    stores HIP/KNEE in bi-space and HAA in serial form.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tau_no_comp = _joint_torque_no_comp_for_cfg(env, asset, asset_cfg)
    return torch.sum(torch.square(tau_no_comp), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize squared joint velocity in bi-space for hip/knee (serial elsewhere)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
    joint_vel = _apply_bispace_state(joint_vel, pairs)
    return torch.sum(torch.square(joint_vel), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize squared joint acceleration in bi-space for hip/knee (serial elsewhere)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
    joint_acc = _apply_bispace_state(joint_acc, pairs)
    return torch.sum(torch.square(joint_acc), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize |q - q_default| in bi-space for hip/knee (serial elsewhere)."""
    asset: Articulation = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
    diff_angle = _apply_bispace_state(diff_angle, pairs)
    return torch.sum(torch.abs(diff_angle), dim=1)


def joint_pos_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 1.0,
    margin_ratio: float = 0.0,
    power: float = 2.0,
    limit_overrides: dict[str, tuple[float, float]] | None = None,
    bispace_limit_overrides: dict[str, tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Penalize joint positions near or beyond soft limits in bi-space for hip/knee.

    Args:
        soft_ratio: Scales the soft interval about its center before applying the penalty.
            `1.0` keeps the asset-provided soft limits unchanged.
        margin_ratio: Fraction of the interval on each side treated as a margin zone.
            `0.0` preserves the old behavior and penalizes only out-of-limit values.
        power: Exponent used for the margin-zone penalty.
        limit_overrides: Optional regex-keyed lower/upper bounds applied before bi-space
            conversion, e.g. `{"FRHIP": (0.6, 1.8), ".*KNEE": (0.2, 2.2)}`.
        bispace_limit_overrides: Optional bi-space lower/upper bounds applied after bi-space
            conversion. Only the keys `QM` and `QB` are used, e.g.
            `{"QM": (0.6, 1.8), "QB": (1.6, 2.3)}`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _selected_joint_ids(asset, asset_cfg)
    joint_names = [str(asset.joint_names[joint_id]) for joint_id in joint_ids]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lower_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    upper_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]

    if limit_overrides:
        lower_limits = lower_limits.clone()
        upper_limits = upper_limits.clone()
        for pattern, bounds in limit_overrides.items():
            regex = re.compile(pattern, re.IGNORECASE)
            override_lower, override_upper = float(bounds[0]), float(bounds[1])
            for idx, joint_name in enumerate(joint_names):
                if regex.fullmatch(joint_name):
                    lower_limits[:, idx] = override_lower
                    upper_limits[:, idx] = override_upper

    pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
    joint_pos = _apply_bispace_state(joint_pos, pairs)
    lower_limits, upper_limits = _apply_bispace_pos_limits(lower_limits, upper_limits, pairs)

    if bispace_limit_overrides and pairs:
        lower_limits = lower_limits.clone()
        upper_limits = upper_limits.clone()
        qm_bounds = bispace_limit_overrides.get("QM") or bispace_limit_overrides.get("qm")
        qb_bounds = bispace_limit_overrides.get("QB") or bispace_limit_overrides.get("qb")
        for hip_idx, knee_idx in pairs:
            if qm_bounds is not None:
                lower_limits[:, hip_idx] = float(qm_bounds[0])
                upper_limits[:, hip_idx] = float(qm_bounds[1])
            if qb_bounds is not None:
                lower_limits[:, knee_idx] = float(qb_bounds[0])
                upper_limits[:, knee_idx] = float(qb_bounds[1])

    center = 0.5 * (lower_limits + upper_limits)
    half_range = 0.5 * torch.clamp(upper_limits - lower_limits, min=1.0e-6)
    scaled_half_range = torch.clamp(half_range * float(soft_ratio), min=1.0e-6)
    effective_lower = center - scaled_half_range
    effective_upper = center + scaled_half_range

    if margin_ratio <= 0.0:
        out_of_limits = torch.clamp(effective_lower - joint_pos, min=0.0)
        out_of_limits += torch.clamp(joint_pos - effective_upper, min=0.0)
        return torch.sum(out_of_limits, dim=1)

    margin = torch.clamp((effective_upper - effective_lower) * float(margin_ratio), min=1.0e-6)
    lower_distance = joint_pos - effective_lower
    upper_distance = effective_upper - joint_pos

    lower_penalty = torch.clamp((margin - lower_distance) / margin, min=0.0)
    upper_penalty = torch.clamp((margin - upper_distance) / margin, min=0.0)
    penalty = torch.pow(lower_penalty, float(power)) + torch.pow(upper_penalty, float(power))
    return torch.sum(penalty, dim=1)




def joint_rom_range_penalty(
    env: ManagerBasedRLEnv,
    q1_lower_limit: float = 0.0,
    q1_upper_limit: float = 1.5707963267948966,
    q2_lower_limit: float = 0.0,
    q2_upper_limit: float = 3.141592653589793,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize HIP/KNEE serial joint-position violations outside configured ROM.

    - q1: HIP in [q1_lower_limit, q1_upper_limit]
    - q2: KNEE in [q2_lower_limit, q2_upper_limit]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _selected_joint_ids(asset, asset_cfg)
    if len(joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    hip_token = str(getattr(env.cfg, "hip_joint_token", "HIP")).upper()
    knee_token = str(getattr(env.cfg, "knee_joint_token", "KNEE")).upper()
    joint_names = [str(asset.joint_names[joint_id]).upper() for joint_id in joint_ids]
    hip_local_ids = [local_idx for local_idx, name in enumerate(joint_names) if hip_token in name]
    knee_local_ids = [local_idx for local_idx, name in enumerate(joint_names) if knee_token in name]

    joint_pos = asset.data.joint_pos[:, joint_ids]
    penalty = torch.zeros(joint_pos.shape[0], device=joint_pos.device, dtype=joint_pos.dtype)
    if len(hip_local_ids) > 0:
        q1 = joint_pos[:, hip_local_ids]
        penalty += torch.sum(torch.clamp(float(q1_lower_limit) - q1, min=0.0), dim=1)
        penalty += torch.sum(torch.clamp(q1 - float(q1_upper_limit), min=0.0), dim=1)
    if len(knee_local_ids) > 0:
        q2 = joint_pos[:, knee_local_ids]
        penalty += torch.sum(torch.clamp(float(q2_lower_limit) - q2, min=0.0), dim=1)
        penalty += torch.sum(torch.clamp(q2 - float(q2_upper_limit), min=0.0), dim=1)
    return penalty


def joint_vel_limits(
    env: ManagerBasedRLEnv,
    soft_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize velocity limit violations in bi-space for hip/knee."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    joint_vel_limits = asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids]

    pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
    joint_vel = _apply_bispace_state(joint_vel, pairs)
    joint_vel_limits = _apply_bispace_abs_limits(joint_vel_limits, pairs)

    out_of_limits = torch.abs(joint_vel) - soft_ratio * joint_vel_limits
    out_of_limits = torch.clamp(out_of_limits, min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)


def applied_torque_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize applied torque values that exceed soft limits in bi-space for hip/knee."""
    asset: Articulation = env.scene[asset_cfg.name]

    applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    if hasattr(asset.data, "soft_joint_torque_limits"):
        soft_torque_limits = asset.data.soft_joint_torque_limits[:, asset_cfg.joint_ids]
    elif hasattr(asset.data, "soft_joint_effort_limits"):
        soft_torque_limits = asset.data.soft_joint_effort_limits[:, asset_cfg.joint_ids]
    elif hasattr(asset.data, "joint_effort_limits"):
        soft_torque_limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    elif getattr(env.cfg, "torque_limit", None) is not None:
        soft_torque_limits = torch.full_like(applied_torque, float(env.cfg.torque_limit))
    else:
        return torch.zeros(applied_torque.shape[0], device=applied_torque.device, dtype=applied_torque.dtype)

    pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
    applied_torque = _apply_bispace_torque(applied_torque, pairs)
    soft_torque_limits = _apply_bispace_abs_limits(soft_torque_limits, pairs)

    out_of_limits = torch.abs(applied_torque) - soft_torque_limits
    return torch.sum(torch.clamp(out_of_limits, min=0.0), dim=1)


def applied_torque_limits_no_comp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    torque_limit: float | None = None,
) -> torch.Tensor:
    """Penalize torque-limit violations excluding tau_comp.

    Torque uses ``tau_no_comp`` when available. The limit is taken from the
    reward parameter when provided, otherwise it falls back to articulation
    soft-effort limits and then ``env.cfg.torque_limit``.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    tau_no_comp = _joint_torque_no_comp_for_cfg(env, asset, asset_cfg)
    if torque_limit is not None:
        soft_torque_limits = torch.full_like(tau_no_comp, float(torque_limit))
        out_of_limits = torch.abs(tau_no_comp) - soft_torque_limits
        return torch.sum(torch.clamp(out_of_limits, min=0.0), dim=1)
    if hasattr(asset.data, "soft_joint_torque_limits"):
        soft_torque_limits = asset.data.soft_joint_torque_limits[:, asset_cfg.joint_ids]
    elif hasattr(asset.data, "soft_joint_effort_limits"):
        soft_torque_limits = asset.data.soft_joint_effort_limits[:, asset_cfg.joint_ids]
    elif hasattr(asset.data, "joint_effort_limits"):
        soft_torque_limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    elif getattr(env.cfg, "torque_limit", None) is not None:
        soft_torque_limits = torch.full_like(tau_no_comp, float(env.cfg.torque_limit))
        out_of_limits = torch.abs(tau_no_comp) - soft_torque_limits
        return torch.sum(torch.clamp(out_of_limits, min=0.0), dim=1)
    else:
        return torch.zeros(tau_no_comp.shape[0], device=tau_no_comp.device, dtype=tau_no_comp.dtype)

    pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
    soft_torque_limits = _apply_bispace_abs_limits(soft_torque_limits, pairs)

    out_of_limits = torch.abs(tau_no_comp) - soft_torque_limits
    return torch.sum(torch.clamp(out_of_limits, min=0.0), dim=1)


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    asset: Articulation = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
    diff_angle = _apply_bispace_state(diff_angle, pairs)
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.06,
) -> torch.Tensor:
    """Penalize joint deviation when command magnitude is small (bi-space for hip/knee)."""
    command = env.command_manager.get_command(command_name)
    return joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :], dim=1) < command_threshold)


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
    diff_angle = _apply_bispace_state(diff_angle, pairs)
    running_reward = torch.linalg.norm(diff_angle, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    pairs = _bi_pairs_for_cfg(env, asset, asset_cfg)
    joint_vel = torch.abs(_apply_bispace_state(joint_vel, pairs))
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    if _use_biarticular_hip_knee(env):
        all_pairs = _resolve_hip_knee_pairs(env, asset, list(range(asset.num_joints)))
        joint_pos = _apply_bispace_state(joint_pos, all_pairs)
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(joint_pos[:, joint_pair[0][0]] - joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    action = env.action_manager.action
    if _use_biarticular_hip_knee(env) and action.shape[1] == asset.num_joints:
        all_pairs = _resolve_hip_knee_pairs(env, asset, list(range(asset.num_joints)))
        action = _apply_bispace_state(action, all_pairs)
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(action[:, joint_pair[0][0]])
                - torch.abs(action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    action = env.action_manager.action
    if _use_biarticular_hip_knee(env) and action.shape[1] == asset.num_joints:
        all_pairs = _resolve_hip_knee_pairs(env, asset, list(range(asset.num_joints)))
        action = _apply_bispace_state(action, all_pairs)

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    """Penalize disagreement across 4-leg joint groups using variance to the group mean."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    if _use_biarticular_hip_knee(env):
        all_pairs = _resolve_hip_knee_pairs(env, asset, list(range(asset.num_joints)))
        joint_pos = _apply_bispace_state(joint_pos, all_pairs)

    if not hasattr(env, "joint_sync_joint_cache") or env.joint_sync_joint_cache is None:
        env.joint_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_group in env.joint_sync_joint_cache:
        if len(joint_group) < 2:
            continue
        joints = torch.stack([joint_pos[:, joint[0]] for joint in joint_group], dim=1)
        group_mean = torch.mean(joints, dim=1, keepdim=True)
        variance = torch.mean(torch.square(joints - group_mean), dim=(1, 2))
        reward += variance

    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# //! Jump Reward
def _foot_contact_mask(contact_sensor: ContactSensor, body_ids, threshold: float) -> torch.Tensor:
    """Return contact mask [num_envs, num_bodies] from force history."""
    net_forces = contact_sensor.data.net_forces_w_history[:, :, body_ids, :]
    return torch.max(torch.norm(net_forces, dim=-1), dim=1)[0] > threshold


# //! Landing Reward
def _touchdown_after_valid_flight(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_air_time: float,
) -> torch.Tensor:
    """True on first touchdown after a sufficiently long flight."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    touchdown = torch.any(first_contact, dim=1)
    valid_flight = torch.max(last_air_time, dim=1)[0] > min_air_time
    return touchdown & valid_flight


# //! Jump Reward
def _cached_body_ids(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    cache_key: str,
    body_names: tuple[str, ...] | list[str],
) -> torch.Tensor:
    """Resolve body names once and reuse ids across calls."""
    if not hasattr(env, "_reward_body_id_cache") or env._reward_body_id_cache is None:
        env._reward_body_id_cache = {}
    key = (cache_key, tuple(body_names))
    if key not in env._reward_body_id_cache:
        body_ids, _ = asset.find_bodies(tuple(body_names), preserve_order=True)
        env._reward_body_id_cache[key] = torch.as_tensor(body_ids, device=asset.device, dtype=torch.long)
    return env._reward_body_id_cache[key]


def _body_ids_tensor(asset: Articulation, body_ids, device: torch.device) -> torch.Tensor:
    if isinstance(body_ids, slice):
        return torch.arange(asset.num_bodies, device=device, dtype=torch.long)
    if torch.is_tensor(body_ids):
        return body_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(body_ids, device=device, dtype=torch.long)


def _body_positions_in_base_frame(asset: Articulation, body_ids: torch.Tensor) -> torch.Tensor:
    body_pos_w = asset.data.body_link_pos_w[:, body_ids, :]
    rel_pos_w = body_pos_w - asset.data.root_link_pos_w[:, None, :]
    quat = asset.data.root_link_quat_w[:, None, :].expand(-1, body_ids.numel(), -1).reshape(-1, 4)
    return quat_apply_inverse(quat, rel_pos_w.reshape(-1, 3)).reshape(asset.data.root_link_pos_w.shape[0], -1, 3)


def object_x_progress(
    env: ManagerBasedRLEnv,
    start_x: float,
    target_x: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("push_object"),
) -> torch.Tensor:
    """Dense 0..1 reward for pushing the object from start_x toward target_x in env frame."""
    push_object: RigidObject = env.scene[object_cfg.name]
    object_x = push_object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    denom = max(float(target_x) - float(start_x), 1.0e-6)
    return torch.clamp((object_x - float(start_x)) / denom, 0.0, 1.0)


def object_target_position_exp(
    env: ManagerBasedRLEnv,
    target_x: float,
    target_y: float,
    x_std: float,
    y_std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("push_object"),
) -> torch.Tensor:
    """Reward the object for reaching the env-relative target xy position."""
    push_object: RigidObject = env.scene[object_cfg.name]
    object_pos_rel = push_object.data.root_pos_w[:, :3] - env.scene.env_origins
    x_std_safe = max(float(x_std), 1.0e-6)
    y_std_safe = max(float(y_std), 1.0e-6)
    err = torch.square(object_pos_rel[:, 0] - float(target_x)) / (x_std_safe**2)
    err += torch.square(object_pos_rel[:, 1] - float(target_y)) / (y_std_safe**2)
    return torch.exp(-err)


def object_forward_velocity(
    env: ManagerBasedRLEnv,
    target_x: float,
    max_velocity: float = 0.6,
    object_cfg: SceneEntityCfg = SceneEntityCfg("push_object"),
) -> torch.Tensor:
    """Reward positive object x velocity while it has not passed the target."""
    push_object: RigidObject = env.scene[object_cfg.name]
    object_x = push_object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    remaining = object_x < float(target_x)
    vel_x = torch.clamp(push_object.data.root_lin_vel_w[:, 0], min=0.0, max=float(max_velocity))
    return vel_x * remaining.float()


def object_y_deviation_l2(
    env: ManagerBasedRLEnv,
    target_y: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("push_object"),
) -> torch.Tensor:
    """Penalize lateral object drift in env frame."""
    push_object: RigidObject = env.scene[object_cfg.name]
    object_y = push_object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return torch.square(object_y - float(target_y))


def root_xy_deviation_l2(
    env: ManagerBasedRLEnv,
    target_x: float = 0.0,
    target_y: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base xy drift in env frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos_rel = asset.data.root_pos_w[:, :3] - env.scene.env_origins
    return torch.square(root_pos_rel[:, 0] - float(target_x)) + torch.square(root_pos_rel[:, 1] - float(target_y))


def fr_wall_normal_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_force: float,
    force_axis: int = 0,
    use_absolute: bool = True,
    force_sign: float = -1.0,
) -> torch.Tensor:
    """Reward normalized FR-foot wall normal force, clipped at the target force."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = torch.arange(contact_sensor.data.net_forces_w.shape[1], device=env.device)
    else:
        body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
    total_forces_w = _contact_sensor_total_forces_w(contact_sensor, body_ids)
    signed_force = total_forces_w[:, :, int(force_axis)].sum(dim=1)
    if bool(use_absolute):
        normal_force = torch.abs(signed_force)
    else:
        normal_force = torch.clamp(float(force_sign) * signed_force, min=0.0)
    return torch.clamp(normal_force / max(float(target_force), 1.0e-6), 0.0, 1.0)


def fr_wall_force_tracking_exp(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_force: float,
    force_std: float,
    force_axis: int = 0,
    use_absolute: bool = True,
    force_sign: float = -1.0,
) -> torch.Tensor:
    """Reward tracking a target wall normal force with the FR foot."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = torch.arange(contact_sensor.data.net_forces_w.shape[1], device=env.device)
    else:
        body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
    total_forces_w = _contact_sensor_total_forces_w(contact_sensor, body_ids)
    signed_force = total_forces_w[:, :, int(force_axis)].sum(dim=1)
    if bool(use_absolute):
        normal_force = torch.abs(signed_force)
    else:
        normal_force = torch.clamp(float(force_sign) * signed_force, min=0.0)
    return torch.exp(-torch.square(normal_force - float(target_force)) / (max(float(force_std), 1.0e-6) ** 2))


def fr_wall_tangential_force_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    normal_axis: int = 0,
) -> torch.Tensor:
    """Penalize tangential FR-foot contact force on the wall."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = torch.arange(contact_sensor.data.net_forces_w.shape[1], device=env.device)
    else:
        body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
    total_forces_w = _contact_sensor_total_forces_w(contact_sensor, body_ids).sum(dim=1)
    axis_ids = [idx for idx in range(3) if idx != int(normal_axis)]
    return torch.sum(torch.square(total_forces_w[:, axis_ids]), dim=1)


def fr_wall_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    saturation_force: float = 5.0,
) -> torch.Tensor:
    """Reward FR-foot contact with the wall using the filtered wall-contact sensor."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = torch.arange(contact_sensor.data.net_forces_w.shape[1], device=env.device)
    else:
        body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
    total_forces_w = _contact_sensor_total_forces_w(contact_sensor, body_ids)
    contact_force = torch.linalg.norm(total_forces_w, dim=-1).sum(dim=1)
    force_range = max(float(saturation_force) - float(threshold), 1.0e-6)
    return torch.clamp((contact_force - float(threshold)) / force_range, 0.0, 1.0)


def fr_wall_x_distance_exp(
    env: ManagerBasedRLEnv,
    wall_contact_x: float,
    std: float,
    foot_body_name: str = "FR_foot",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reducing the base-frame x distance between FR foot and the wall plane."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_ids = _cached_body_ids(env, asset, "_fr_wall_x_distance", (foot_body_name,))
    foot_pos_rel = asset.data.body_link_pos_w[:, foot_ids[0], :] - env.scene.env_origins
    distance_x = torch.clamp(float(wall_contact_x) - foot_pos_rel[:, 0], min=0.0)
    return torch.exp(-torch.square(distance_x) / (max(float(std), 1.0e-6) ** 2))


def fr_foot_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    foot_body_name: str = "FR_foot",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward FR foot height close to a target world height above flat ground."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_ids = _cached_body_ids(env, asset, "_fr_foot_height", (foot_body_name,))
    foot_z_rel = asset.data.body_link_pos_w[:, foot_ids[0], 2] - env.scene.env_origins[:, 2]
    return torch.exp(-torch.square(foot_z_rel - float(target_height)) / (max(float(std), 1.0e-6) ** 2))


def fr_foot_body_wall_xz_exp(
    env: ManagerBasedRLEnv,
    wall_contact_x: float,
    target_z_b: float,
    std: float,
    foot_body_name: str = "FR_foot",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wall_cfg: SceneEntityCfg | None = None,
    wall_half_x: float = 0.0,
) -> torch.Tensor:
    """Reward FR foot x-z position reaching the wall target in the base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_ids = _cached_body_ids(env, asset, "_fr_foot_body_wall_xz", (foot_body_name,))

    root_pos_w = asset.data.root_link_pos_w
    root_quat_w = asset.data.root_link_quat_w
    foot_pos_w = asset.data.body_link_pos_w[:, foot_ids[0], :]
    foot_pos_b = math_utils.quat_apply_inverse(root_quat_w, foot_pos_w - root_pos_w)

    wall_point_w = root_pos_w.clone()
    if wall_cfg is not None:
        wall: RigidObject = env.scene[wall_cfg.name]
        wall_point_w[:, 0] = wall.data.root_pos_w[:, 0] - float(wall_half_x)
    else:
        wall_point_w[:, 0] = env.scene.env_origins[:, 0] + float(wall_contact_x)
    wall_pos_b = math_utils.quat_apply_inverse(root_quat_w, wall_point_w - root_pos_w)

    target_xz_b = torch.stack(
        (wall_pos_b[:, 0], torch.full_like(wall_pos_b[:, 0], float(target_z_b))),
        dim=1,
    )
    foot_xz_b = foot_pos_b[:, [0, 2]]
    return torch.exp(-torch.linalg.norm(foot_xz_b - target_xz_b, dim=1) / max(float(std), 1.0e-6))


def fr_wall_command_force_tracking_exp(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_force: float,
    force_std: float,
    force_axis: int = 0,
    use_absolute: bool = True,
    force_sign: float = -1.0,
) -> torch.Tensor:
    """Reward matching the commanded wall normal force with sim wall-contact force."""
    return fr_wall_force_tracking_exp(
        env=env,
        sensor_cfg=sensor_cfg,
        target_force=float(command_force),
        force_std=float(force_std),
        force_axis=int(force_axis),
        use_absolute=bool(use_absolute),
        force_sign=float(force_sign),
    )


def selected_feet_ground_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward selected feet maintaining ground contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = _foot_contact_mask(contact_sensor, sensor_cfg.body_ids, float(threshold))
    return torch.mean(contact.float(), dim=1)


def selected_feet_ground_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize selected feet touching the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = _foot_contact_mask(contact_sensor, sensor_cfg.body_ids, float(threshold))
    return torch.mean(contact.float(), dim=1)


def support_feet_env_position_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    axes: str = "xy",
) -> torch.Tensor:
    """Penalize selected support feet moving away from their episode-start env-frame positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = _body_ids_tensor(asset, asset_cfg.body_ids, env.device)
    foot_pos_rel = asset.data.body_link_pos_w[:, body_ids, :] - env.scene.env_origins[:, None, :]

    cache_name = "_fr_push_support_foot_targets"
    if not hasattr(env, cache_name) or getattr(env, cache_name) is None:
        setattr(env, cache_name, foot_pos_rel.detach().clone())
    target_pos = getattr(env, cache_name)
    if target_pos.shape != foot_pos_rel.shape:
        target_pos = foot_pos_rel.detach().clone()
        setattr(env, cache_name, target_pos)

    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        reset_like = env.episode_length_buf <= 1
        if torch.any(reset_like):
            target_pos[reset_like] = foot_pos_rel[reset_like].detach()

    axis_ids = [0, 1] if axes == "xy" else [0, 1, 2]
    return torch.sum(torch.square(foot_pos_rel[:, :, axis_ids] - target_pos[:, :, axis_ids]), dim=(1, 2))


def fr_foot_push_pose_exp(
    env: ManagerBasedRLEnv,
    object_local_offset: tuple[float, float, float],
    xy_std: float,
    z_std: float,
    foot_body_name: str = "FR_foot",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("push_object"),
) -> torch.Tensor:
    """Reward FR foot near a desired local point on the object, typically its rear face."""
    asset: Articulation = env.scene[asset_cfg.name]
    push_object: RigidObject = env.scene[object_cfg.name]
    foot_ids = _cached_body_ids(env, asset, "_fr_push_pose", (foot_body_name,))

    foot_pos_w = asset.data.body_link_pos_w[:, foot_ids[0], :]
    offset_local = torch.tensor(object_local_offset, device=env.device, dtype=foot_pos_w.dtype).unsqueeze(0)
    offset_w = math_utils.quat_apply(push_object.data.root_quat_w, offset_local.expand(env.num_envs, -1))
    target_pos_w = push_object.data.root_pos_w[:, :3] + offset_w

    diff = foot_pos_w - target_pos_w
    xy_std_safe = max(float(xy_std), 1.0e-6)
    z_std_safe = max(float(z_std), 1.0e-6)
    err = torch.sum(torch.square(diff[:, :2]), dim=1) / (xy_std_safe**2)
    err += torch.square(diff[:, 2]) / (z_std_safe**2)
    return torch.exp(-err)


def support_feet_body_position_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize selected support feet leaving their initial base-frame stance."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = _body_ids_tensor(asset, asset_cfg.body_ids, env.device)
    foot_pos_b = _body_positions_in_base_frame(asset, body_ids)

    cache_name = "_fr_push_support_foot_body_targets"
    if not hasattr(env, cache_name) or getattr(env, cache_name) is None:
        setattr(env, cache_name, foot_pos_b.detach().clone())
    target_pos = getattr(env, cache_name)
    if target_pos.shape != foot_pos_b.shape:
        target_pos = foot_pos_b.detach().clone()
        setattr(env, cache_name, target_pos)

    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        reset_like = env.episode_length_buf <= 1
        if torch.any(reset_like):
            target_pos[reset_like] = foot_pos_b[reset_like].detach()

    return torch.sum(torch.square(foot_pos_b - target_pos), dim=(1, 2))


# //! Landing Reward
def _feet_on_top_mask(
    foot_pos_w: torch.Tensor,
    target_x_w: torch.Tensor,
    target_y_w: torch.Tensor,
    top_half_x: float,
    top_half_y: float,
    edge_margin: float,
    top_height: float,
    z_tol: float,
) -> torch.Tensor:
    """Per-foot boolean mask indicating support inside shrunken top surface."""
    safe_half_x = max(float(top_half_x) - float(edge_margin), 1e-4)
    safe_half_y = max(float(top_half_y) - float(edge_margin), 1e-4)
    inside_x = torch.abs(foot_pos_w[:, :, 0] - target_x_w.unsqueeze(1)) <= safe_half_x
    inside_y = torch.abs(foot_pos_w[:, :, 1] - target_y_w.unsqueeze(1)) <= safe_half_y
    above_top = foot_pos_w[:, :, 2] >= (float(top_height) - float(z_tol))
    return inside_x & inside_y & above_top


# //! Jump Reward
def all_feet_air(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    command_name: str | None = None,
    active_when_jump: bool = True,
) -> torch.Tensor:
    """Reward when all selected feet are airborne."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = _foot_contact_mask(contact_sensor, sensor_cfg.body_ids, threshold)
    reward = (contact.sum(dim=1) == 0).float()
    return reward * _command_phase_mask(env, command_name, active_when_jump)


# //! Jump Reward
def jump_crouch_prep_exp(
    env: ManagerBasedRLEnv,
    target_crouch_z: float,
    std: float,
    sensor_cfg: SceneEntityCfg,
    min_contact_feet: int = 2,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_when_jump: bool = True,
) -> torch.Tensor:
    """Reward squat-like prep posture while in stance before takeoff."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = _foot_contact_mask(contact_sensor, sensor_cfg.body_ids, threshold)
    stance = contact.sum(dim=1) >= int(min_contact_feet)
    z_abs = asset.data.root_pos_w[:, 2]
    std_safe = max(float(std), 1e-6)
    reward = stance.float() * torch.exp(-torch.square(z_abs - float(target_crouch_z)) / (std_safe**2))
    return reward * _command_phase_mask(env, command_name, active_when_jump)


# //! Jump Reward
def jump_obstacle_clearance_exp(
    env: ManagerBasedRLEnv,
    top_height: float,
    margin: float,
    std: float,
    body_names: tuple[str, ...] | list[str],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_when_jump: bool = True,
) -> torch.Tensor:
    """Reward sufficient body-point clearance above obstacle top during flight."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = _foot_contact_mask(contact_sensor, sensor_cfg.body_ids, threshold)
    airborne = (contact.sum(dim=1) == 0).float()
    body_ids = _cached_body_ids(env, asset, "_jump_clearance", tuple(body_names))
    body_pos_w = asset.data.body_link_pos_w[:, body_ids, :]
    deficit = torch.clamp(float(top_height + margin) - body_pos_w[:, :, 2], min=0.0)
    std_safe = max(float(std), 1e-6)
    clearance_score = torch.exp(-torch.square(deficit) / (std_safe**2)).mean(dim=1)
    reward = airborne * clearance_score
    return reward * _command_phase_mask(env, command_name, active_when_jump)


# //! Landing Reward
def landing_prediction_xy_exp(
    env: ManagerBasedRLEnv,
    target_x: float,
    target_y: float,
    target_z: float,
    std_xy: float,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    gravity: float = 9.81,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_when_jump: bool = True,
) -> torch.Tensor:
    """Reward ballistic landing-point prediction matching target top location."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = _foot_contact_mask(contact_sensor, sensor_cfg.body_ids, threshold)
    airborne = contact.sum(dim=1) == 0

    pos_w = asset.data.root_pos_w
    vel_w = asset.data.root_lin_vel_w
    z0 = pos_w[:, 2]
    vz = vel_w[:, 2]
    g = max(float(gravity), 1e-6)

    disc = torch.square(vz) - 2.0 * g * (float(target_z) - z0)
    valid = disc > 0.0
    t_hit = (vz + torch.sqrt(torch.clamp(disc, min=0.0))) / g
    t_hit = torch.clamp(t_hit, min=0.0)

    pred_x = pos_w[:, 0] + vel_w[:, 0] * t_hit
    pred_y = pos_w[:, 1] + vel_w[:, 1] * t_hit

    target_x_w = env.scene.env_origins[:, 0] + float(target_x)
    target_y_w = env.scene.env_origins[:, 1] + float(target_y)
    err_sq = torch.square(pred_x - target_x_w) + torch.square(pred_y - target_y_w)
    std_safe = max(float(std_xy), 1e-6)
    reward = (airborne & valid).float() * torch.exp(-err_sq / (std_safe**2))
    return reward * _command_phase_mask(env, command_name, active_when_jump)


# //! Jump Reward
def root_x_progress(
    env: ManagerBasedRLEnv,
    target_x: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_when_jump: bool = True,
) -> torch.Tensor:
    """Dense progress reward on env-relative root x-position."""
    asset: RigidObject = env.scene[asset_cfg.name]
    x_rel = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    denom = max(float(target_x), 1e-3)
    reward = torch.clamp(x_rel / denom, 0.0, 1.0)
    return reward * _command_phase_mask(env, command_name, active_when_jump)


# //! Landing Reward
def touchdown_target_x_exp(
    env: ManagerBasedRLEnv,
    target_x: float,
    std: float,
    min_air_time: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_when_jump: bool = True,
) -> torch.Tensor:
    """Reward touchdown landing near target x after a valid flight."""
    asset: RigidObject = env.scene[asset_cfg.name]
    touchdown = _touchdown_after_valid_flight(env, sensor_cfg, min_air_time).float()
    x_rel = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    err = x_rel - target_x
    std_safe = max(float(std), 1e-6)
    reward = touchdown * torch.exp(-torch.square(err) / (std_safe**2))
    return reward * _command_phase_mask(env, command_name, active_when_jump)


# //! Landing Reward
def landing_stable_exp(
    env: ManagerBasedRLEnv,
    min_air_time: float,
    lin_vel_z_std: float,
    ang_vel_xy_std: float,
    ori_std: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_when_jump: bool = True,
) -> torch.Tensor:
    """Reward stable touchdown: low vertical speed, low pitch/roll rate, upright base."""
    asset: RigidObject = env.scene[asset_cfg.name]
    touchdown = _touchdown_after_valid_flight(env, sensor_cfg, min_air_time).float()

    vz = torch.square(asset.data.root_lin_vel_b[:, 2])
    ang_xy = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    ori = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

    lv_std = max(float(lin_vel_z_std), 1e-6)
    av_std = max(float(ang_vel_xy_std), 1e-6)
    ori_std_safe = max(float(ori_std), 1e-6)

    score = torch.exp(-(vz / (lv_std**2) + ang_xy / (av_std**2) + ori / (ori_std_safe**2)))
    reward = touchdown * score
    return reward * _command_phase_mask(env, command_name, active_when_jump)


# //! Landing Reward
def block_top_pose_exp(
    env: ManagerBasedRLEnv,
    target_x: float,
    target_z: float,
    x_std: float,
    z_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_when_jump: bool = True,
) -> torch.Tensor:
    """Reward root pose near desired block-top landing pose (env-relative x, world z)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    x_rel = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    z_abs = asset.data.root_pos_w[:, 2]
    x_err = x_rel - float(target_x)
    z_err = z_abs - float(target_z)
    x_std_safe = max(float(x_std), 1e-6)
    z_std_safe = max(float(z_std), 1e-6)
    reward = torch.exp(-(torch.square(x_err) / (x_std_safe**2) + torch.square(z_err) / (z_std_safe**2)))
    return reward * _command_phase_mask(env, command_name, active_when_jump)


# //! Landing Reward
def jump_success_hold_bonus(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    foot_body_names: tuple[str, str, str, str],
    top_center_x: float,
    top_center_y: float,
    top_height: float,
    top_half_x: float,
    top_half_y: float,
    edge_margin: float = 0.02,
    hold_time_s: float = 0.20,
    lin_vel_threshold: float = 0.25,
    ang_vel_threshold: float = 1.5,
    z_tol: float = 0.02,
    contact_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_when_jump: bool = True,
    switch_to_stand_after_success: bool = False,
    switch_command_name: str | None = None,
    stand_value: int = 0,
) -> torch.Tensor:
    """Sparse success bonus when stable support on top is held for a minimum duration."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_ids = _cached_body_ids(env, asset, "_jump_success_feet", foot_body_names)

    if not hasattr(env, "_jump_success_hold_time") or env._jump_success_hold_time is None:
        env._jump_success_hold_time = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_jump_success_awarded") or env._jump_success_awarded is None:
        env._jump_success_awarded = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    if hasattr(env, "reset_buf"):
        reset_ids = torch.nonzero(env.reset_buf, as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            env._jump_success_hold_time[reset_ids] = 0.0
            env._jump_success_awarded[reset_ids] = False

    foot_pos_w = asset.data.body_link_pos_w[:, foot_ids, :]
    target_x_w = env.scene.env_origins[:, 0] + float(top_center_x)
    target_y_w = env.scene.env_origins[:, 1] + float(top_center_y)
    on_top = _feet_on_top_mask(
        foot_pos_w=foot_pos_w,
        target_x_w=target_x_w,
        target_y_w=target_y_w,
        top_half_x=top_half_x,
        top_half_y=top_half_y,
        edge_margin=edge_margin,
        top_height=top_height,
        z_tol=z_tol,
    )

    contact = _foot_contact_mask(contact_sensor, sensor_cfg.body_ids, contact_threshold)
    all_support = torch.all(on_top, dim=1) & torch.all(contact, dim=1)
    stable = (
        torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1) < float(lin_vel_threshold)
    ) & (torch.linalg.norm(asset.data.root_ang_vel_b[:, :3], dim=1) < float(ang_vel_threshold))
    good = all_support & stable

    env._jump_success_hold_time = torch.where(
        good, env._jump_success_hold_time + float(env.step_dt), torch.zeros_like(env._jump_success_hold_time)
    )
    just_success = (env._jump_success_hold_time >= float(hold_time_s)) & (~env._jump_success_awarded)
    env._jump_success_awarded = env._jump_success_awarded | just_success
    phase_mask = _command_phase_mask(env, command_name, active_when_jump)
    reward = just_success.float() * phase_mask

    # Optional mode switch: after successful landing+hold, change task command to stand mode.
    if bool(switch_to_stand_after_success) and switch_command_name and torch.any(just_success):
        try:
            term = env.command_manager.get_term(switch_command_name)
            if hasattr(term, "command_buffer"):
                term.command_buffer[just_success] = int(stand_value)
        except Exception:
            pass
    return reward


# //! Landing Reward
def jump_fail_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    support_sensor_cfg: SceneEntityCfg,
    foot_body_names: tuple[str, str, str, str],
    top_center_x: float,
    top_center_y: float,
    top_height: float,
    top_half_x: float,
    top_half_y: float,
    edge_margin: float = 0.02,
    min_base_height: float = 0.12,
    orientation_limit: float = 0.8,
    z_tol: float = 0.02,
    contact_threshold: float = 1.0,
    body_contact_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_when_jump: bool = True,
) -> torch.Tensor:
    """Fail indicator: body collision, collapse, bad orientation, or falling off after top touch."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_contact_sensor: ContactSensor = env.scene.sensors[support_sensor_cfg.name]
    foot_ids = _cached_body_ids(env, asset, "_jump_fail_feet", foot_body_names)

    if not hasattr(env, "_jump_touched_top") or env._jump_touched_top is None:
        env._jump_touched_top = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if hasattr(env, "reset_buf"):
        reset_ids = torch.nonzero(env.reset_buf, as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            env._jump_touched_top[reset_ids] = False

    body_collision = _foot_contact_mask(body_contact_sensor, sensor_cfg.body_ids, body_contact_threshold).any(dim=1)
    base_low = asset.data.root_pos_w[:, 2] < float(min_base_height)
    orientation_bad = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) > float(orientation_limit)

    foot_pos_w = asset.data.body_link_pos_w[:, foot_ids, :]
    target_x_w = env.scene.env_origins[:, 0] + float(top_center_x)
    target_y_w = env.scene.env_origins[:, 1] + float(top_center_y)
    on_top = _feet_on_top_mask(
        foot_pos_w=foot_pos_w,
        target_x_w=target_x_w,
        target_y_w=target_y_w,
        top_half_x=top_half_x,
        top_half_y=top_half_y,
        edge_margin=edge_margin,
        top_height=top_height,
        z_tol=z_tol,
    )
    foot_contact = _foot_contact_mask(foot_contact_sensor, support_sensor_cfg.body_ids, contact_threshold)
    top_touch_now = torch.any(on_top & foot_contact, dim=1)
    env._jump_touched_top = env._jump_touched_top | top_touch_now
    fell_off_after_touch = env._jump_touched_top & (~torch.any(on_top, dim=1)) & (
        asset.data.root_pos_w[:, 2] < float(top_height - 0.03)
    )

    fail = body_collision | base_low | orientation_bad | fell_off_after_touch
    reward = fail.float()
    return reward * _command_phase_mask(env, command_name, active_when_jump)


# //! Jump Reward
def aerial_tuck_exp(
    env: ManagerBasedRLEnv,
    target_dist: float,
    std: float,
    front_body_names: tuple[str, str],
    hind_body_names: tuple[str, str],
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward near-touch tuck in flight using front/hind sagittal distance."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = _foot_contact_mask(contact_sensor, sensor_cfg.body_ids, threshold)
    airborne = (contact.sum(dim=1) == 0).float()

    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "_aerial_tuck_body_cache") or env._aerial_tuck_body_cache is None:
        front_ids, _ = asset.find_bodies(front_body_names, preserve_order=True)
        hind_ids, _ = asset.find_bodies(hind_body_names, preserve_order=True)
        env._aerial_tuck_body_cache = (front_ids, hind_ids)
    else:
        front_ids, hind_ids = env._aerial_tuck_body_cache

    front_pos = asset.data.body_link_pos_w[:, front_ids, :].mean(dim=1)
    hind_pos = asset.data.body_link_pos_w[:, hind_ids, :].mean(dim=1)
    delta_w = front_pos - hind_pos
    delta_b = quat_apply_inverse(asset.data.root_link_quat_w, delta_w)

    # Use sagittal plane distance (x-z) in body frame.
    sagittal_dist = torch.linalg.norm(delta_b[:, [0, 2]], dim=1)
    std_safe = max(float(std), 1e-6)
    return airborne * torch.exp(-torch.square(sagittal_dist - target_dist) / (std_safe**2))


# def feet_air_time(
#     env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
# ) -> torch.Tensor:
#     """Reward long steps taken by the feet using L2-kernel.

#     This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
#     that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
#     the time for which the feet are in the air.

#     If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
#     """
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     # print(torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1), "command norm")
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    # return reward

# def feet_air_time(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     sensor_cfg: SceneEntityCfg,
#     mode_time: float,
#     velocity_threshold: float,
# ) -> torch.Tensor:
#     """Reward longer feet air and contact time."""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     asset: Articulation = env.scene[asset_cfg.name]
#     if contact_sensor.cfg.track_air_time is False:
#         raise RuntimeError("Activate ContactSensor's track_air_time!")
#     # compute the reward
#     current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
#     current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

#     t_max = torch.max(current_air_time, current_contact_time)
#     t_min = torch.clip(t_max, max=mode_time)
#     stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
#     cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
#     body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
#     reward = torch.where(
#         torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
#         torch.where(t_max < mode_time, t_min, 0),
#         stance_cmd_reward,
#     )
#     return torch.sum(reward, dim=1)


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1)
    # print(last_air_time, "last air time")
    # print(last_contact_time, "last contact time")
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward




def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.5
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_count_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    expect_contact_num: int,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize deviation from the desired number of currently contacting feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = _foot_contact_mask(contact_sensor, sensor_cfg.body_ids, threshold)
    contact_num = torch.sum(contact.int(), dim=1)
    reward = torch.abs(contact_num - int(expect_contact_num)).float()
    reward *= (torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > float(command_threshold)).float()
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # print(contact, "contact")
    reward = torch.sum(contact, dim=-1).float()
    # print(reward, "reward after sum")
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.5
    # print(env.command_manager.get_command(command_name), "env.command_manager.get_command(command_name)")
    # print(reward, "reward after multiply")
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_ratio(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward maintaining continuous foot contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = _foot_contact_mask(contact_sensor, sensor_cfg.body_ids, threshold)
    return torch.mean(contact.float(), dim=1)


def feet_stuck_time_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    max_mode_time: float = 0.35,
    max_excess_time: float = 0.5,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize feet that stay too long in contact or in air while a move command is active."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    mode_time = torch.maximum(air_time, contact_time)
    excess_time = torch.clamp(mode_time - float(max_mode_time), min=0.0, max=float(max_excess_time))
    reward = torch.mean(excess_time, dim=1)
    reward *= (torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > float(command_threshold)).float()
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    # foot_velocity_tanh = torch.tanh(
    #     tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    # )
    # reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward = torch.sum(foot_z_target_error, dim=1)
    # print(foot_z_target_error, "foot_z_target_error")
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.2
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# def stand_still_joint_deviation_l1(
#     env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize offsets from the default joint positions when the command is very small."""
    # command = env.command_manager.get_command(command_name)
#     # Penalize motion when command is nearly zero.
#     return joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :], dim=1) < command_threshold)

# def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize joint positions that deviate from the default one."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # compute out of limits constraints
#     angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
#     return torch.sum(torch.abs(angle), dim=1)


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def joint_acc_l2_new(env: ManagerBasedRLEnv) -> torch.Tensor:

# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     # print(torch.sum(diff, dim=1), "smoothness l2")
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward





def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_x_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize x-axis base linear velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 0])


def lin_vel_y_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize y-axis base linear velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_x_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize x-axis base angular velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 0])


def ang_x_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Alias of x-axis base angular-velocity penalty using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 0])


def ang_vel_y_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize y-axis base angular velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 1])


def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base angular velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_air_time_including_ang_z(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > float(command_threshold)
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward

def lin_vel_xy_l2_with_ang_z_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Penalize xy-axis base linear velocity using L2 squared kernel if command is ang_vel_z."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    command = env.command_manager.get_command(command_name)
    reward *= (torch.sum(torch.square(command[:, 2:]), dim=1) > command_threshold) & \
            (torch.sum(torch.square(command[:, :2]), dim=1) < command_threshold)
    # reward *= torch.sum(torch.square(env.command_manager.get_command(command_name)[:, 2:]), dim=1) > command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward



def spi_jump_reach_x_target(
    env: ManagerBasedRLEnv,
    target_x: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """SPI-Active style forward-jump root-x target reward."""
    asset: RigidObject = env.scene[asset_cfg.name]
    x_diff = torch.abs(_jump_root_x_rel(env, asset) - float(target_x))
    return torch.exp(-x_diff / max(float(std), 1.0e-6))


def spi_jump_reach_z_target(
    env: ManagerBasedRLEnv,
    target_z: float,
    start_s: float = 0.75,
    end_s: float = 1.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """SPI-Active style vertical target error during the flight window."""
    asset: RigidObject = env.scene[asset_cfg.name]
    z_diff = torch.abs(asset.data.root_pos_w[:, 2] - float(target_z))
    return z_diff * _jump_time_mask(env, start_s, end_s)


def spi_jump_feet_height_before_jump(
    env: ManagerBasedRLEnv,
    std: float,
    ground_clearance: float = 0.02,
    pre_end_s: float = 0.75,
    post_start_s: float = 1.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward keeping feet low before takeoff and after landing, matching SPI-Active's shaping."""
    asset: RigidObject = env.scene[asset_cfg.name]
    current_time = _jump_time(env)
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].reshape(env.num_envs, -1) - float(ground_clearance)
    stance_window = torch.logical_or(current_time < float(pre_end_s), current_time > float(post_start_s))
    reward = torch.exp(-torch.clamp(feet_height, min=0.0).sum(dim=1) / max(float(std), 1.0e-6))
    return reward * stance_window.to(dtype=reward.dtype)


def spi_jump_lin_vel_z(
    env: ManagerBasedRLEnv,
    start_s: float = 0.5,
    end_s: float = 1.0,
    max_vel: float = 1.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward upward base velocity during takeoff."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_b = getattr(asset.data, "root_lin_vel_b", asset.data.root_lin_vel_w)
    return torch.clamp(lin_vel_b[:, 2], max=float(max_vel)) * _jump_time_mask(env, start_s, end_s)


def spi_jump_lin_vel_x(
    env: ManagerBasedRLEnv,
    start_s: float = 0.75,
    end_s: float = 1.35,
    max_vel: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward base velocity during forward-jump flight."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_b = getattr(asset.data, "root_lin_vel_b", asset.data.root_lin_vel_w)
    return torch.clamp(lin_vel_b[:, 0], max=float(max_vel)) * _jump_time_mask(env, start_s, end_s)


def spi_jump_height_control(
    env: ManagerBasedRLEnv,
    target_height: float,
    start_s: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base height after the jump sequence."""
    asset: RigidObject = env.scene[asset_cfg.name]
    current_time = _jump_time(env)
    return torch.square(float(target_height) - asset.data.root_pos_w[:, 2]) * (current_time > float(start_s)).float()


def spi_jump_actions_symmetry(
    env: ManagerBasedRLEnv,
    std: float,
) -> torch.Tensor:
    """SPI-Active action symmetry reward for left/right leg pairs."""
    action = env.action_manager.action
    if action.shape[1] < 12:
        return torch.zeros(env.num_envs, device=env.device)
    diff = torch.square(action[:, 0] + action[:, 3])
    diff += torch.square(action[:, 1:3] - action[:, 4:6]).sum(dim=-1)
    diff += torch.square(action[:, 6] + action[:, 9])
    diff += torch.square(action[:, 7:9] - action[:, 10:12]).sum(dim=-1)
    return torch.exp(-diff / max(float(std), 1.0e-6))


def spi_jump_penalty_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from the initial upright projected-gravity direction."""
    asset: RigidObject = env.scene[asset_cfg.name]
    desired_projected_gravity = torch.zeros_like(asset.data.projected_gravity_b)
    desired_projected_gravity[:, 2] = -1.0
    return torch.sum(torch.square(asset.data.projected_gravity_b - desired_projected_gravity), dim=1)


def spi_jump_penalty_slippage(
    env: ManagerBasedRLEnv,
    pre_end_s: float = 0.75,
    post_start_s: float = 1.2,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize total foot velocity before takeoff and after landing."""
    asset: RigidObject = env.scene[asset_cfg.name]
    current_time = _jump_time(env)
    active = torch.logical_or(current_time > float(post_start_s), current_time < float(pre_end_s))
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    return torch.sum(torch.linalg.norm(foot_vel, dim=-1), dim=1) * active.float()


def spi_jump_penalty_contact_during_air(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    start_s: float = 0.75,
    end_s: float = 1.4,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize any foot contact during the intended airborne window."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = torch.any(_foot_contact_mask(contact_sensor, sensor_cfg.body_ids, float(threshold)), dim=1)
    return contact.float() * _jump_time_mask(env, start_s, end_s)


def spi_jump_penalty_contact_landing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    start_s: float = 0.9,
    end_s: float = 1.25,
) -> torch.Tensor:
    """Penalize summed foot contact-force norm in the landing transition window."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = torch.arange(contact_sensor.data.net_forces_w.shape[1], device=env.device)
    else:
        body_ids = torch.as_tensor(body_ids, device=env.device)
    forces = _contact_sensor_total_forces_w(contact_sensor, body_ids)
    contact_norm = torch.linalg.norm(torch.linalg.norm(forces, dim=2), dim=1)
    return contact_norm * _jump_time_mask(env, start_s, end_s)


def spi_jump_landing_contact_ratio(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    start_s: float = 1.45,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward recovering foot contact after the intended airborne phase."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = _foot_contact_mask(contact_sensor, sensor_cfg.body_ids, float(threshold))
    active = (_jump_time(env) > float(start_s)).to(dtype=torch.float32)
    return torch.mean(contacts.to(dtype=torch.float32), dim=1) * active


def spi_jump_feet_distance(
    env: ManagerBasedRLEnv,
    stance_width: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral foot placement error in body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos_b = _jump_feet_in_body_frame(env, asset, asset_cfg.body_ids)
    desired_y = torch.tensor([0.5, -0.5, 0.5, -0.5], device=env.device, dtype=foot_pos_b.dtype)
    desired_y = desired_y[: foot_pos_b.shape[1]].unsqueeze(0) * float(stance_width)
    return torch.square(desired_y - foot_pos_b[:, :, 1]).sum(dim=1)


def spi_jump_feet_x(
    env: ManagerBasedRLEnv,
    default_stance_length: float = 0.1934,
    extra_stance_length: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize fore-aft foot placement error in body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos_b = _jump_feet_in_body_frame(env, asset, asset_cfg.body_ids)
    desired_x = torch.tensor([1.0, 1.0, -1.0, -1.0], device=env.device, dtype=foot_pos_b.dtype)
    desired_x = desired_x[: foot_pos_b.shape[1]].unsqueeze(0) * float(default_stance_length)
    desired_x = desired_x + float(extra_stance_length)
    return torch.abs(desired_x - foot_pos_b[:, :, 0]).sum(dim=1)



# //! Landing Reward
def touchdown_lateral_drift_exp(
    env: ManagerBasedRLEnv,
    std: float,
    min_air_time: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_when_jump: bool = True,
) -> torch.Tensor:
    """Reward touchdown with minimal env-relative lateral drift after a valid flight."""
    asset: RigidObject = env.scene[asset_cfg.name]
    touchdown = _touchdown_after_valid_flight(env, sensor_cfg, min_air_time).float()
    y_rel = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    std_safe = max(float(std), 1e-6)
    reward = touchdown * torch.exp(-torch.square(y_rel) / (std_safe**2))
    return reward * _command_phase_mask(env, command_name, active_when_jump)


# //! Landing Reward
def jump_apex_height_touchdown_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    min_air_time: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_when_jump: bool = True,
) -> torch.Tensor:
    """Reward the achieved peak height at touchdown, measured relative to the episode start height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    current_z = asset.data.root_pos_w[:, 2]

    if not hasattr(env, "_jump_init_root_height") or env._jump_init_root_height is None:
        env._jump_init_root_height = current_z.clone()
    if not hasattr(env, "_jump_peak_root_height") or env._jump_peak_root_height is None:
        env._jump_peak_root_height = current_z.clone()

    if hasattr(env, "reset_buf"):
        reset_ids = torch.nonzero(env.reset_buf, as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            env._jump_init_root_height[reset_ids] = current_z[reset_ids]
            env._jump_peak_root_height[reset_ids] = current_z[reset_ids]

    env._jump_peak_root_height = torch.maximum(env._jump_peak_root_height, current_z)

    touchdown = _touchdown_after_valid_flight(env, sensor_cfg, min_air_time).float()
    achieved_height = env._jump_peak_root_height - env._jump_init_root_height
    std_safe = max(float(std), 1e-6)
    reward = touchdown * torch.exp(-torch.square(achieved_height - float(target_height)) / (std_safe**2))
    return reward * _command_phase_mask(env, command_name, active_when_jump)


# //! Landing Reward
def landing_stable_exp(
    env: ManagerBasedRLEnv,
    min_air_time: float,
    lin_vel_z_std: float,
    ang_vel_xy_std: float,
    ori_std: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_when_jump: bool = True,
) -> torch.Tensor:
    """Reward stable touchdown: low vertical speed, low pitch/roll rate, upright base."""
    asset: RigidObject = env.scene[asset_cfg.name]
    touchdown = _touchdown_after_valid_flight(env, sensor_cfg, min_air_time).float()

    vz = torch.square(asset.data.root_lin_vel_b[:, 2])
    ang_xy = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    ori = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

    lv_std = max(float(lin_vel_z_std), 1e-6)
    av_std = max(float(ang_vel_xy_std), 1e-6)
    ori_std_safe = max(float(ori_std), 1e-6)

    score = torch.exp(-(vz / (lv_std**2) + ang_xy / (av_std**2) + ori / (ori_std_safe**2)))
    reward = touchdown * score
    return reward * _command_phase_mask(env, command_name, active_when_jump)


# //! Landing Reward
def block_top_pose_exp(
    env: ManagerBasedRLEnv,
    target_x: float,
    target_z: float,
    x_std: float,
    z_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_when_jump: bool = True,
) -> torch.Tensor:
    """Reward root pose near desired block-top landing pose (env-relative x, world z)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    x_rel = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    z_abs = asset.data.root_pos_w[:, 2]
    x_err = x_rel - float(target_x)
    z_err = z_abs - float(target_z)
    x_std_safe = max(float(x_std), 1e-6)
    z_std_safe = max(float(z_std), 1e-6)
    reward = torch.exp(-(torch.square(x_err) / (x_std_safe**2) + torch.square(z_err) / (z_std_safe**2)))
    return reward * _command_phase_mask(env, command_name, active_when_jump)


# //! Landing Reward
def jump_success_hold_bonus(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    foot_body_names: tuple[str, str, str, str],
    top_center_x: float,
    top_center_y: float,
    top_height: float,
    top_half_x: float,
    top_half_y: float,
    edge_margin: float = 0.02,
    hold_time_s: float = 0.20,
    lin_vel_threshold: float = 0.25,
    ang_vel_threshold: float = 1.5,
    z_tol: float = 0.02,
    contact_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_when_jump: bool = True,
    switch_to_stand_after_success: bool = False,
    switch_command_name: str | None = None,
    stand_value: int = 0,
) -> torch.Tensor:
    """Sparse success bonus when stable support on top is held for a minimum duration."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_ids = _cached_body_ids(env, asset, "_jump_success_feet", foot_body_names)

    if not hasattr(env, "_jump_success_hold_time") or env._jump_success_hold_time is None:
        env._jump_success_hold_time = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_jump_success_awarded") or env._jump_success_awarded is None:
        env._jump_success_awarded = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    if hasattr(env, "reset_buf"):
        reset_ids = torch.nonzero(env.reset_buf, as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            env._jump_success_hold_time[reset_ids] = 0.0
            env._jump_success_awarded[reset_ids] = False

    foot_pos_w = asset.data.body_link_pos_w[:, foot_ids, :]
    target_x_w = env.scene.env_origins[:, 0] + float(top_center_x)
    target_y_w = env.scene.env_origins[:, 1] + float(top_center_y)
    on_top = _feet_on_top_mask(
        foot_pos_w=foot_pos_w,
        target_x_w=target_x_w,
        target_y_w=target_y_w,
        top_half_x=top_half_x,
        top_half_y=top_half_y,
        edge_margin=edge_margin,
        top_height=top_height,
        z_tol=z_tol,
    )

    contact = _foot_contact_mask(contact_sensor, sensor_cfg.body_ids, contact_threshold)
    all_support = torch.all(on_top, dim=1) & torch.all(contact, dim=1)
    stable = (
        torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1) < float(lin_vel_threshold)
    ) & (torch.linalg.norm(asset.data.root_ang_vel_b[:, :3], dim=1) < float(ang_vel_threshold))
    good = all_support & stable

    env._jump_success_hold_time = torch.where(
        good, env._jump_success_hold_time + float(env.step_dt), torch.zeros_like(env._jump_success_hold_time)
    )
    just_success = (env._jump_success_hold_time >= float(hold_time_s)) & (~env._jump_success_awarded)
    env._jump_success_awarded = env._jump_success_awarded | just_success
    phase_mask = _command_phase_mask(env, command_name, active_when_jump)
    reward = just_success.float() * phase_mask

    # Optional mode switch: after successful landing+hold, change task command to stand mode.
    if bool(switch_to_stand_after_success) and switch_command_name and torch.any(just_success):
        try:
            term = env.command_manager.get_term(switch_command_name)
            if hasattr(term, "command_buffer"):
                term.command_buffer[just_success] = int(stand_value)
        except Exception:
            pass
    return reward


# //! Landing Reward
def jump_fail_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    support_sensor_cfg: SceneEntityCfg,
    foot_body_names: tuple[str, str, str, str],
    top_center_x: float,
    top_center_y: float,
    top_height: float,
    top_half_x: float,
    top_half_y: float,
    edge_margin: float = 0.02,
    min_base_height: float = 0.12,
    orientation_limit: float = 0.8,
    z_tol: float = 0.02,
    contact_threshold: float = 1.0,
    body_contact_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_when_jump: bool = True,
) -> torch.Tensor:
    """Fail indicator: body collision, collapse, bad orientation, or falling off after top touch."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_contact_sensor: ContactSensor = env.scene.sensors[support_sensor_cfg.name]
    foot_ids = _cached_body_ids(env, asset, "_jump_fail_feet", foot_body_names)

    if not hasattr(env, "_jump_touched_top") or env._jump_touched_top is None:
        env._jump_touched_top = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if hasattr(env, "reset_buf"):
        reset_ids = torch.nonzero(env.reset_buf, as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            env._jump_touched_top[reset_ids] = False

    body_collision = _foot_contact_mask(body_contact_sensor, sensor_cfg.body_ids, body_contact_threshold).any(dim=1)
    base_low = asset.data.root_pos_w[:, 2] < float(min_base_height)
    orientation_bad = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) > float(orientation_limit)

    foot_pos_w = asset.data.body_link_pos_w[:, foot_ids, :]
    target_x_w = env.scene.env_origins[:, 0] + float(top_center_x)
    target_y_w = env.scene.env_origins[:, 1] + float(top_center_y)
    on_top = _feet_on_top_mask(
        foot_pos_w=foot_pos_w,
        target_x_w=target_x_w,
        target_y_w=target_y_w,
        top_half_x=top_half_x,
        top_half_y=top_half_y,
        edge_margin=edge_margin,
        top_height=top_height,
        z_tol=z_tol,
    )
    foot_contact = _foot_contact_mask(foot_contact_sensor, support_sensor_cfg.body_ids, contact_threshold)
    top_touch_now = torch.any(on_top & foot_contact, dim=1)
    env._jump_touched_top = env._jump_touched_top | top_touch_now
    fell_off_after_touch = env._jump_touched_top & (~torch.any(on_top, dim=1)) & (
        asset.data.root_pos_w[:, 2] < float(top_height - 0.03)
    )

    fail = body_collision | base_low | orientation_bad | fell_off_after_touch
    reward = fail.float()
    return reward * _command_phase_mask(env, command_name, active_when_jump)


def _jump_root_x_rel(env: ManagerBasedRLEnv, asset: RigidObject | Articulation) -> torch.Tensor:
    return asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]


def _jump_time_mask(env: ManagerBasedRLEnv, start_s: float, end_s: float) -> torch.Tensor:
    current_time = _jump_time(env)
    return torch.logical_and(current_time > float(start_s), current_time < float(end_s)).to(dtype=torch.float32)


def _jump_time(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.episode_length_buf.to(dtype=torch.float32) * float(env.step_dt)

def _jump_feet_in_body_frame(
    env: ManagerBasedRLEnv,
    asset: RigidObject | Articulation,
    body_ids,
) -> torch.Tensor:
    foot_pos_w = asset.data.body_pos_w[:, body_ids, :]
    rel_pos_w = foot_pos_w - asset.data.root_pos_w[:, :3].unsqueeze(1)
    n_feet = rel_pos_w.shape[1]
    return math_utils.quat_apply_inverse(
        asset.data.root_quat_w[:, None, :].expand(-1, n_feet, -1).reshape(-1, 4),
        rel_pos_w.reshape(-1, 3),
    ).reshape(env.num_envs, n_feet, 3)

# //! Jump Reward
def aerial_tuck_exp(
    env: ManagerBasedRLEnv,
    target_dist: float,
    std: float,
    front_body_names: tuple[str, str],
    hind_body_names: tuple[str, str],
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward near-touch tuck in flight using front/hind sagittal distance."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = _foot_contact_mask(contact_sensor, sensor_cfg.body_ids, threshold)
    airborne = (contact.sum(dim=1) == 0).float()

    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "_aerial_tuck_body_cache") or env._aerial_tuck_body_cache is None:
        front_ids, _ = asset.find_bodies(front_body_names, preserve_order=True)
        hind_ids, _ = asset.find_bodies(hind_body_names, preserve_order=True)
        env._aerial_tuck_body_cache = (front_ids, hind_ids)
    else:
        front_ids, hind_ids = env._aerial_tuck_body_cache

    front_pos = asset.data.body_link_pos_w[:, front_ids, :].mean(dim=1)
    hind_pos = asset.data.body_link_pos_w[:, hind_ids, :].mean(dim=1)
    delta_w = front_pos - hind_pos
    delta_b = quat_apply_inverse(asset.data.root_link_quat_w, delta_w)

    # Use sagittal plane distance (x-z) in body frame.
    sagittal_dist = torch.linalg.norm(delta_b[:, [0, 2]], dim=1)
    std_safe = max(float(std), 1e-6)
    return airborne * torch.exp(-torch.square(sagittal_dist - target_dist) / (std_safe**2))
