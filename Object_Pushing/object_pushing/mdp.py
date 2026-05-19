from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg, RewardTermCfg
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom, Vt

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


OBJECT_DIAMETER = 0.4
OBJECT_RADIUS = 0.5 * OBJECT_DIAMETER
OBJECT_HEIGHT = 0.2
OBJECT_CENTER_Z = 0.5 * OBJECT_HEIGHT
OBJECT_DIAMETER_RANGE = (0.4, 0.5)
OBJECT_HEIGHT_RANGE = (0.2, 0.3)
OBJECT_MASS = 2.0
ROBOT_RADIUS_START_RANGE = (0.6, 1.0)
TARGET_DISTANCE_START_RANGE = (0.3, 0.8)
ROBOT_RADIUS_END_RANGE = (0.6, 3.0)
TARGET_DISTANCE_END_RANGE = (0.3, 3.0)
ROBOT_LATERAL_START_RANGE = (-0.05, 0.05)
ROBOT_LATERAL_END_RANGE = (-3.0, 3.0)
ROBOT_YAW_NOISE_START_RANGE = (-0.10, 0.10)
ROBOT_YAW_NOISE_END_RANGE = (-math.pi, math.pi)
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


def _set_root_scale(prim_path: str, scale: tuple[float, float, float]) -> None:
    stage = get_current_stage()
    prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
    scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
    if scale_spec is None:
        scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)
        op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
        if op_order_spec is None:
            op_order_spec = Sdf.AttributeSpec(
                prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
            )
        op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
    scale_spec.default = Gf.Vec3f(*scale)


def randomize_pushing_object_geometry(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    diameter_range: tuple[float, float] = OBJECT_DIAMETER_RANGE,
    height_range: tuple[float, float] = OBJECT_HEIGHT_RANGE,
    object_cfg: SceneEntityCfg = SceneEntityCfg("push_object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
) -> None:
    """Randomize object and target-marker cylinder scale before physics starts."""
    if env.sim.is_playing():
        raise RuntimeError("Object geometry randomization must run in the prestartup event mode.")

    if env_ids is None:
        env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids_cpu = env_ids.cpu()

    object_paths = sim_utils.find_matching_prim_paths(env.scene[object_cfg.name].cfg.prim_path)
    target_paths = sim_utils.find_matching_prim_paths(env.scene[target_cfg.name].cfg.prim_path)

    diameters = torch.empty(len(env_ids_cpu), device="cpu").uniform_(diameter_range[0], diameter_range[1])
    heights = torch.empty(len(env_ids_cpu), device="cpu").uniform_(height_range[0], height_range[1])
    xy_scales = diameters / float(OBJECT_DIAMETER)
    z_scales = heights / float(OBJECT_HEIGHT)

    if not hasattr(env, "_object_pushing_object_heights"):
        env._object_pushing_object_heights = torch.full(
            (env.scene.num_envs,), float(OBJECT_HEIGHT), device=env.device
        )
        env._object_pushing_object_diameters = torch.full(
            (env.scene.num_envs,), float(OBJECT_DIAMETER), device=env.device
        )

    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids_cpu.tolist()):
            scale = (float(xy_scales[i]), float(xy_scales[i]), float(z_scales[i]))
            _set_root_scale(object_paths[env_id], scale)
            _set_root_scale(target_paths[env_id], scale)
            env._object_pushing_object_heights[env_id] = float(heights[i])
            env._object_pushing_object_diameters[env_id] = float(diameters[i])


def object_pushing_spawn_distance_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    steps_per_iteration: int = 40,
    iterations_per_level: int = 100,
    num_levels: int = 20,
    robot_radius_start_range: tuple[float, float] = ROBOT_RADIUS_START_RANGE,
    robot_radius_end_range: tuple[float, float] = ROBOT_RADIUS_END_RANGE,
    target_distance_start_range: tuple[float, float] = TARGET_DISTANCE_START_RANGE,
    target_distance_end_range: tuple[float, float] = TARGET_DISTANCE_END_RANGE,
    robot_lateral_start_range: tuple[float, float] = ROBOT_LATERAL_START_RANGE,
    robot_lateral_end_range: tuple[float, float] = ROBOT_LATERAL_END_RANGE,
    robot_yaw_noise_start_range: tuple[float, float] = ROBOT_YAW_NOISE_START_RANGE,
    robot_yaw_noise_end_range: tuple[float, float] = ROBOT_YAW_NOISE_END_RANGE,
) -> dict[str, float]:
    """Increase object-robot and object-target spawn distance every fixed PPO iteration interval."""
    del env_ids
    level_interval_steps = max(1, int(steps_per_iteration) * int(iterations_per_level))
    level = min(int(env.common_step_counter // level_interval_steps), int(num_levels))
    alpha = float(level) / float(max(1, num_levels))

    robot_range = tuple(
        float(start + alpha * (end - start))
        for start, end in zip(robot_radius_start_range, robot_radius_end_range, strict=True)
    )
    target_range = tuple(
        float(start + alpha * (end - start))
        for start, end in zip(target_distance_start_range, target_distance_end_range, strict=True)
    )
    lateral_range = tuple(
        float(start + alpha * (end - start))
        for start, end in zip(robot_lateral_start_range, robot_lateral_end_range, strict=True)
    )
    yaw_noise_range = tuple(
        float(start + alpha * (end - start))
        for start, end in zip(robot_yaw_noise_start_range, robot_yaw_noise_end_range, strict=True)
    )

    env._object_pushing_robot_radius_range = robot_range
    env._object_pushing_target_distance_range = target_range
    env._object_pushing_robot_lateral_range = lateral_range
    env._object_pushing_robot_yaw_noise_range = yaw_noise_range

    return {
        "level": float(level),
        "robot_min": robot_range[0],
        "robot_max": robot_range[1],
        "target_min": target_range[0],
        "target_max": target_range[1],
        "lateral_abs_max": max(abs(lateral_range[0]), abs(lateral_range[1])),
        "yaw_noise_abs_max": max(abs(yaw_noise_range[0]), abs(yaw_noise_range[1])),
    }


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
    robot_radius_range: tuple[float, float] = ROBOT_RADIUS_START_RANGE,
    object_xy_range: tuple[float, float] = (-0.5, 0.5),
    object_yaw_range: tuple[float, float] = (-math.pi, math.pi),
    target_distance_range: tuple[float, float] = TARGET_DISTANCE_START_RANGE,
    target_angle_range: tuple[float, float] = (-math.pi, math.pi),
    robot_lateral_range: tuple[float, float] = ROBOT_LATERAL_START_RANGE,
    robot_yaw_noise_range: tuple[float, float] = ROBOT_YAW_NOISE_START_RANGE,
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
    robot_radius_range = getattr(env, "_object_pushing_robot_radius_range", robot_radius_range)
    target_distance_range = getattr(env, "_object_pushing_target_distance_range", target_distance_range)
    robot_lateral_range = getattr(env, "_object_pushing_robot_lateral_range", robot_lateral_range)
    robot_yaw_noise_range = getattr(env, "_object_pushing_robot_yaw_noise_range", robot_yaw_noise_range)
    object_heights = getattr(env, "_object_pushing_object_heights", None)
    if object_heights is None:
        object_center_z = torch.full((count, 1), OBJECT_CENTER_Z, device=device)
    else:
        object_center_z = 0.5 * object_heights[env_ids].unsqueeze(1)

    object_xy = torch.empty(count, 2, device=device).uniform_(object_xy_range[0], object_xy_range[1])
    object_yaw = torch.empty(count, device=device).uniform_(object_yaw_range[0], object_yaw_range[1])
    target_dist = torch.empty(count, device=device).uniform_(target_distance_range[0], target_distance_range[1])
    target_angle = torch.empty(count, device=device).uniform_(target_angle_range[0], target_angle_range[1])
    target_dir = torch.stack((torch.cos(target_angle), torch.sin(target_angle)), dim=1)
    lateral_dir = torch.stack((-torch.sin(target_angle), torch.cos(target_angle)), dim=1)
    target_xy = object_xy + target_dist.unsqueeze(1) * target_dir

    robot_radius = torch.empty(count, device=device).uniform_(robot_radius_range[0], robot_radius_range[1])
    robot_lateral_sample = torch.empty(count, device=device).uniform_(robot_lateral_range[0], robot_lateral_range[1])
    max_lateral = 0.95 * robot_radius
    robot_lateral = torch.clamp(robot_lateral_sample, -max_lateral, max_lateral)
    robot_back_distance = torch.sqrt(torch.clamp(robot_radius.square() - robot_lateral.square(), min=0.0))
    robot_xy = object_xy - robot_back_distance.unsqueeze(1) * target_dir + robot_lateral.unsqueeze(1) * lateral_dir
    robot_yaw_noise = torch.empty(count, device=device).uniform_(robot_yaw_noise_range[0], robot_yaw_noise_range[1])
    robot_yaw = target_angle + robot_yaw_noise

    robot_pos = origins + torch.cat((robot_xy, torch.full((count, 1), BASE_HEIGHT, device=device)), dim=1)
    object_pos = origins + torch.cat((object_xy, object_center_z), dim=1)
    target_pos = origins + torch.cat((target_xy, object_center_z), dim=1)

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


def front_push_reward(
    env: "ManagerBasedRLEnv",
    centerline_std: float = 0.25,
    front_distance: float = 0.8,
    front_distance_std: float = 0.5,
    max_forward_speed_reward: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("push_object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    push_object: RigidObject = env.scene[object_cfg.name]

    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w
    object_pos_w = push_object.data.root_pos_w
    object_vel_w = _object_lin_vel(push_object)

    robot_to_object_b = math_utils.quat_apply_inverse(robot_quat_w, object_pos_w - robot_pos_w)
    robot_forward_w = math_utils.quat_apply(robot_quat_w, _unit_x(env.device).repeat(env.num_envs, 1))
    forward_speed = torch.sum(robot_forward_w[:, :2] * object_vel_w[:, :2], dim=1)

    front_gate = (robot_to_object_b[:, 0] > 0.0).float()
    center_gate = torch.exp(-torch.square(robot_to_object_b[:, 1] / float(centerline_std)))
    distance_gate = torch.exp(-torch.relu(robot_to_object_b[:, 0] - float(front_distance)) / float(front_distance_std))
    speed_reward = torch.clamp(torch.relu(forward_speed), max=float(max_forward_speed_reward))
    return front_gate * center_gate * distance_gate * speed_reward


def target_facing_reward(
    env: "ManagerBasedRLEnv",
    centerline_std: float = 0.35,
    proximity_std: float = 1.5,
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

    robot_to_object_b = math_utils.quat_apply_inverse(robot_quat_w, object_pos_w - robot_pos_w)
    robot_to_object_dist = torch.norm((object_pos_w - robot_pos_w)[:, :2], dim=1)
    object_to_target_dir = _normalize_xy(target_pos_w - object_pos_w)
    robot_forward_w = math_utils.quat_apply(robot_quat_w, _unit_x(env.device).repeat(env.num_envs, 1))
    target_alignment = torch.sum(robot_forward_w[:, :2] * object_to_target_dir, dim=1)

    front_gate = (robot_to_object_b[:, 0] > 0.0).float()
    center_gate = torch.exp(-torch.square(robot_to_object_b[:, 1] / float(centerline_std)))
    proximity_gate = torch.exp(-robot_to_object_dist / float(proximity_std))
    return front_gate * center_gate * proximity_gate * torch.square(torch.relu(target_alignment))


class intrinsic_pushing_reward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.previous_action = torch.zeros(env.num_envs, 3, device=env.device)
        self.previous_previous_action = torch.zeros(env.num_envs, 3, device=env.device)
        self.previous_intrinsic = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        k1: float = 2.0,
        k3: float = 3.0,
        k4: float = 2.0,
        k5: float = 0.2,
        k6: float = 0.2,
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
        r_i3 = float(k3) * torch.clamp(torch.relu(forward_speed), max=1.0)
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
    position_threshold: float = 0.15,
    object_cfg: SceneEntityCfg = SceneEntityCfg("push_object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
) -> torch.Tensor:
    push_object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    pos_error = torch.norm((push_object.data.root_pos_w - target.data.root_pos_w)[:, :2], dim=1)
    return pos_error < float(position_threshold)


def success_bonus(env: "ManagerBasedRLEnv", bonus: float = 1000.0) -> torch.Tensor:
    return object_goal_success(env).float() * float(bonus)


def base_below_height(
    env: "ManagerBasedRLEnv",
    minimum_height: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < float(minimum_height)
