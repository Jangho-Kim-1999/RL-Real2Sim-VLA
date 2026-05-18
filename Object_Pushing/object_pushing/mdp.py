from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg, RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


OBJECT_DIAMETER = 0.4
OBJECT_RADIUS = 0.5 * OBJECT_DIAMETER
OBJECT_HEIGHT = 0.2
OBJECT_CENTER_Z = 0.5 * OBJECT_HEIGHT
BASE_HEIGHT = 0.35
FOOT_PREFIX_ORDER = ("FL", "FR", "RL", "RR")


def _unit_x(device: str | torch.device) -> torch.Tensor:
    return torch.tensor((1.0, 0.0, 0.0), device=device)


def _normalize_xy(vec: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    xy = vec[:, :2]
    return xy / torch.clamp(torch.norm(xy, dim=1, keepdim=True), min=eps)


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    half_yaw = 0.5 * yaw
    return torch.stack(
        (torch.cos(half_yaw), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(half_yaw)),
        dim=-1,
    )


def _asset_pos_quat(asset: RigidObject | Articulation) -> tuple[torch.Tensor, torch.Tensor]:
    return asset.data.root_pos_w, asset.data.root_quat_w


def _object_lin_vel(asset: RigidObject) -> torch.Tensor:
    if hasattr(asset.data, "root_lin_vel_w"):
        return asset.data.root_lin_vel_w
    return asset.data.root_vel_w[:, :3]


def _sort_foot_ids(asset: Articulation, foot_ids: Sequence[int]) -> list[int]:
    names = [asset.body_names[int(body_id)] for body_id in foot_ids]
    sorted_ids: list[int] = []
    for prefix in FOOT_PREFIX_ORDER:
        for body_id, name in zip(foot_ids, names, strict=False):
            if name.upper().startswith(prefix) and int(body_id) not in sorted_ids:
                sorted_ids.append(int(body_id))
                break
    for body_id in foot_ids:
        if int(body_id) not in sorted_ids:
            sorted_ids.append(int(body_id))
    return sorted_ids[:4]


class pushing_policy_observation(ManagerTermBase):
    """33-D high-level position-only pushing observation in the robot base frame."""

    def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        robot: Articulation = env.scene[cfg.params.get("robot_cfg", SceneEntityCfg("robot")).name]
        foot_body_names = cfg.params.get("foot_body_names", ".*foot")
        foot_ids, _ = robot.find_bodies(foot_body_names, preserve_order=True)
        self.foot_body_ids = _sort_foot_ids(robot, foot_ids)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("push_object"),
        target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
        action_name: str = "joint_pos",
        foot_body_names: str = ".*foot",
    ) -> torch.Tensor:
        del foot_body_names
        robot: Articulation = env.scene[robot_cfg.name]
        push_object: RigidObject = env.scene[object_cfg.name]
        target: RigidObject = env.scene[target_cfg.name]

        robot_pos_w, robot_quat_w = _asset_pos_quat(robot)
        object_pos_w, _ = _asset_pos_quat(push_object)
        target_pos_w, _ = _asset_pos_quat(target)

        x_axis = _unit_x(env.device).repeat(env.num_envs, 1)
        robot_forward_w = math_utils.quat_apply(robot_quat_w, x_axis)

        feet_rel_b = torch.zeros(env.num_envs, 4, 3, device=env.device)
        if self.foot_body_ids:
            feet_pos_w = robot.data.body_link_pos_w[:, self.foot_body_ids, :]
            feet_rel_w = feet_pos_w - robot_pos_w.unsqueeze(1)
            feet_rel_b[:, : feet_pos_w.shape[1], :] = math_utils.quat_apply_inverse(
                robot_quat_w[:, None, :].expand(-1, feet_pos_w.shape[1], -1).reshape(-1, 4),
                feet_rel_w.reshape(-1, 3),
            ).reshape(env.num_envs, feet_pos_w.shape[1], 3)

        robot_to_object_b = math_utils.quat_apply_inverse(robot_quat_w, object_pos_w - robot_pos_w)
        object_to_target_b = math_utils.quat_apply_inverse(robot_quat_w, target_pos_w - object_pos_w)
        robot_to_target_b = math_utils.quat_apply_inverse(robot_quat_w, target_pos_w - robot_pos_w)

        object_vel_b = math_utils.quat_apply_inverse(robot_quat_w, _object_lin_vel(push_object))

        action_term = env.action_manager.get_term(action_name)
        last_action = action_term.velocity_command

        return torch.cat(
            (
                robot_forward_w,
                robot.data.root_ang_vel_b,
                feet_rel_b.reshape(env.num_envs, 12),
                _normalize_xy(robot_to_object_b),
                torch.norm(robot_to_object_b[:, :2], dim=1, keepdim=True),
                _normalize_xy(object_to_target_b),
                torch.norm(object_to_target_b[:, :2], dim=1, keepdim=True),
                _normalize_xy(robot_to_target_b),
                torch.norm(robot_to_target_b[:, :2], dim=1, keepdim=True),
                object_vel_b,
                last_action,
            ),
            dim=1,
        )


def reset_pushing_scene(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    robot_radius_range: tuple[float, float] = (2.0, 4.0),
    object_xy_range: tuple[float, float] = (-0.5, 0.5),
    target_distance_range: tuple[float, float] = (0.5, 1.5),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("push_object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
):
    robot: Articulation = env.scene[robot_cfg.name]
    push_object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    count = len(env_ids)
    device = env.device
    origins = env.scene.env_origins[env_ids]

    object_xy = torch.empty(count, 2, device=device).uniform_(object_xy_range[0], object_xy_range[1])
    object_yaw = torch.empty(count, device=device).uniform_(-math.pi, math.pi)
    target_dist = torch.empty(count, device=device).uniform_(target_distance_range[0], target_distance_range[1])
    target_angle = torch.empty(count, device=device).uniform_(-math.pi, math.pi)
    target_xy = object_xy + target_dist.unsqueeze(1) * torch.stack(
        (torch.cos(target_angle), torch.sin(target_angle)), dim=1
    )

    robot_radius = torch.empty(count, device=device).uniform_(robot_radius_range[0], robot_radius_range[1])
    robot_angle = torch.empty(count, device=device).uniform_(-math.pi, math.pi)
    robot_xy = object_xy + robot_radius.unsqueeze(1) * torch.stack(
        (torch.cos(robot_angle), torch.sin(robot_angle)), dim=1
    )
    robot_yaw = torch.empty(count, device=device).uniform_(-math.pi, math.pi)

    robot_pos = origins + torch.cat((robot_xy, torch.full((count, 1), BASE_HEIGHT, device=device)), dim=1)
    object_pos = origins + torch.cat((object_xy, torch.full((count, 1), OBJECT_CENTER_Z, device=device)), dim=1)
    target_pos = origins + torch.cat((target_xy, torch.full((count, 1), OBJECT_CENTER_Z, device=device)), dim=1)

    robot.write_root_pose_to_sim(torch.cat((robot_pos, _yaw_quat(robot_yaw)), dim=1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(count, 6, device=device), env_ids=env_ids)
    push_object.write_root_pose_to_sim(torch.cat((object_pos, _yaw_quat(object_yaw)), dim=1), env_ids=env_ids)
    push_object.write_root_velocity_to_sim(torch.zeros(count, 6, device=device), env_ids=env_ids)
    target_quat = _yaw_quat(torch.zeros(count, device=device))
    target.write_root_pose_to_sim(torch.cat((target_pos, target_quat), dim=1), env_ids=env_ids)
    target.write_root_velocity_to_sim(torch.zeros(count, 6, device=device), env_ids=env_ids)


def object_position_reward(
    env: "ManagerBasedRLEnv",
    k2: float = 4.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("push_object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
) -> torch.Tensor:
    push_object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    pos_error = torch.norm((push_object.data.root_pos_w - target.data.root_pos_w)[:, :2], dim=1)
    return -float(k2) * torch.log(pos_error + 0.05)


class intrinsic_pushing_reward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.previous_action = torch.zeros(env.num_envs, 3, device=env.device)
        self.previous_previous_action = torch.zeros(env.num_envs, 3, device=env.device)
        self.previous_intrinsic = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        k1: float = 1.0,
        k3: float = 1.0,
        k4: float = 1.0,
        k5: float = 2.0,
        k6: float = 3.0,
        k7: float = 1.0,
        switch_distance: float = 0.2,
        action_name: str = "joint_pos",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("push_object"),
        target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
    ) -> torch.Tensor:
        robot: Articulation = env.scene[robot_cfg.name]
        push_object: RigidObject = env.scene[object_cfg.name]
        target: RigidObject = env.scene[target_cfg.name]

        robot_pos_w = robot.data.root_pos_w
        robot_quat_w = robot.data.root_quat_w
        object_pos_w = push_object.data.root_pos_w
        target_pos_w = target.data.root_pos_w
        object_vel_w = _object_lin_vel(push_object)

        robot_to_object_dist = torch.norm((object_pos_w - robot_pos_w)[:, :2], dim=1)
        object_to_target = target_pos_w - object_pos_w
        object_to_target_dist = torch.norm(object_to_target[:, :2], dim=1)
        robot_forward_w = math_utils.quat_apply(robot_quat_w, _unit_x(env.device).repeat(env.num_envs, 1))
        forward_speed = torch.sum(robot_forward_w[:, :2] * object_vel_w[:, :2], dim=1)
        target_speed = torch.sum(_normalize_xy(object_to_target) * object_vel_w[:, :2], dim=1)

        action = env.action_manager.get_term(action_name).velocity_command
        r_i1 = float(k1) * torch.exp(-robot_to_object_dist)
        r_i3 = float(k3) * torch.exp(-torch.relu(-forward_speed))
        r_i4 = float(k4) * torch.exp(-torch.relu(-target_speed))
        r_i5 = float(k5) * torch.exp(-torch.norm(action - self.previous_action, dim=1))
        action_jerk = action - 2.0 * self.previous_action + self.previous_previous_action
        r_i6 = float(k6) * torch.exp(-torch.norm(action_jerk, dim=1))
        intrinsic = float(k7) * (r_i1 + r_i3 + r_i4 + r_i5 + r_i6)
        intrinsic = torch.where(object_to_target_dist < float(switch_distance), self.previous_intrinsic, intrinsic)

        self.previous_previous_action[:] = self.previous_action
        self.previous_action[:] = action
        self.previous_intrinsic[:] = intrinsic
        return intrinsic

    def reset(self, env_ids: torch.Tensor):
        self.previous_action[env_ids] = 0.0
        self.previous_previous_action[env_ids] = 0.0
        self.previous_intrinsic[env_ids] = 0.0


def object_goal_success(
    env: "ManagerBasedRLEnv",
    position_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("push_object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
) -> torch.Tensor:
    push_object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    pos_error = torch.norm((push_object.data.root_pos_w - target.data.root_pos_w)[:, :2], dim=1)
    return pos_error < float(position_threshold)


def success_bonus(env: "ManagerBasedRLEnv", bonus: float = 100.0) -> torch.Tensor:
    return object_goal_success(env).float() * float(bonus)


def base_below_height(
    env: "ManagerBasedRLEnv",
    minimum_height: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < float(minimum_height)
