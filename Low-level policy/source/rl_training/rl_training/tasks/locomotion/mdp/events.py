# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Literal, Sequence

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign only the selected env/body subset to avoid shape mismatch when body_ids is not all bodies.
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias[env_ids[:, None], body_ids]

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the center of mass (COM) positions for the rigid bodies.

    This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
    randomized by adding, scaling, or setting random values sampled from the specified distribution.

    .. tip::
        This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor | None): Specific environment indices to apply randomization, or None for all environments.
        asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
        com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
        operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
        distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
    """
    # Extract the asset (Articulation or RigidObject)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Resolve environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current COM offsets (num_assets, num_bodies, 3)
    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):  # Randomize x, y, z independently
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # Set the randomized COM offsets into the simulation
    asset.root_physx_view.set_coms(com_offsets, env_ids)


def reset_root_state_uniform_with_foot_projection(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    min_foot_clearance: float = 0.01,
    foot_body_names: str | Sequence[str] = (".*_foot", ".*FOOT"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset root state uniformly and lift base z to keep feet above terrain origin height.

    This performs a kinematic projection step:
    1. Sample root pose/velocity as in ``reset_root_state_uniform``.
    2. Estimate foot positions for the sampled orientation using current foot offsets in root frame.
    3. Lift root ``z`` so the lowest estimated foot is at least ``env_origin_z + min_foot_clearance``.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # sample root pose around default state
    root_states = asset.data.default_root_state[env_ids].clone()
    pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_ranges = torch.tensor(pose_range_list, device=asset.device)
    rand_pose = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=asset.device
    )

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_pose[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_pose[:, 3], rand_pose[:, 4], rand_pose[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # compute base z lift from foot projection (if foot bodies are found)
    foot_body_ids, _ = asset.find_bodies(foot_body_names, preserve_order=True)
    if len(foot_body_ids) > 0:
        # Current foot offsets in root frame (captures reset joint randomization if executed earlier in reset order).
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
        positions[:, 2] += z_lift

    # sample root velocities
    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=asset.device)
    rand_vel = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_vel

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_rigid_object_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("push_object"),
):
    """Reset a rigid object root state around its default env-relative pose."""
    asset: RigidObject = env.scene[asset_cfg.name]

    root_states = asset.data.default_root_state[env_ids].clone()
    pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_ranges = torch.tensor(pose_range_list, device=asset.device)
    rand_pose = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=asset.device
    )

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_pose[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_pose[:, 3], rand_pose[:, 4], rand_pose[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=asset.device)
    rand_vel = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_vel

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_hip_knee_joints_by_bispace_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    qm_range: tuple[float, float],
    qb_range: tuple[float, float],
    velocity_range: tuple[float, float] = (0.0, 0.0),
    hip_joint_names: str | Sequence[str] = ".*HIP",
    knee_joint_names: str | Sequence[str] = ".*KNEE",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset hip/knee joints using bi-space randomization.

    Bi-space variables are defined as:
    - qm = q_hip
    - qb = q_hip + q_knee

    On reset, this term samples offsets in bi-space and maps back to serial joints:
    - q_hip = q_hip0 + delta_qm
    - q_knee = (q_hip0 + q_knee0 + delta_qb) - q_hip
    """
    asset: Articulation = env.scene[asset_cfg.name]

    hip_ids, hip_names = asset.find_joints(hip_joint_names, preserve_order=True)
    knee_ids, knee_names = asset.find_joints(knee_joint_names, preserve_order=True)
    if len(hip_ids) == 0 or len(knee_ids) == 0:
        return

    def _prefix_from_name(name: str, token: str) -> str:
        name_upper = name.upper()
        token_upper = token.upper()
        token_pos = name_upper.find(token_upper)
        if token_pos >= 0:
            return name[:token_pos]
        return name.rsplit("_", 1)[0] if "_" in name else name

    knee_by_prefix: dict[str, int] = {}
    for knee_id, knee_name in zip(knee_ids, knee_names):
        knee_by_prefix[_prefix_from_name(knee_name, "KNEE")] = int(knee_id)

    pairs: list[tuple[int, int]] = []
    for hip_id, hip_name in zip(hip_ids, hip_names):
        knee_id = knee_by_prefix.get(_prefix_from_name(hip_name, "HIP"))
        if knee_id is not None:
            pairs.append((int(hip_id), int(knee_id)))

    if len(pairs) == 0:
        return

    pairs.sort(key=lambda pair: pair[0])
    hip_joint_ids = [pair[0] for pair in pairs]
    knee_joint_ids = [pair[1] for pair in pairs]
    hip_joint_ids_tensor = torch.tensor(hip_joint_ids, dtype=torch.long, device=asset.device)
    knee_joint_ids_tensor = torch.tensor(knee_joint_ids, dtype=torch.long, device=asset.device)

    # Default state for selected envs.
    default_joint_pos = asset.data.default_joint_pos[env_ids]
    default_joint_vel = asset.data.default_joint_vel[env_ids]

    q_hip_0 = default_joint_pos[:, hip_joint_ids_tensor]
    q_knee_0 = default_joint_pos[:, knee_joint_ids_tensor]
    qb_0 = q_hip_0 + q_knee_0

    # Sample offsets in bi-space.
    delta_qm = math_utils.sample_uniform(*qm_range, q_hip_0.shape, device=asset.device)
    delta_qb = math_utils.sample_uniform(*qb_range, q_hip_0.shape, device=asset.device)

    q_hip = q_hip_0 + delta_qm
    q_knee = (qb_0 + delta_qb) - q_hip

    qd_hip = default_joint_vel[:, hip_joint_ids_tensor]
    qd_knee = default_joint_vel[:, knee_joint_ids_tensor]
    qd_hip += math_utils.sample_uniform(*velocity_range, qd_hip.shape, device=asset.device)
    qd_knee += math_utils.sample_uniform(*velocity_range, qd_knee.shape, device=asset.device)

    # Clamp to soft limits before writing to sim.
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]

    q_hip = q_hip.clamp_(joint_pos_limits[:, hip_joint_ids_tensor, 0], joint_pos_limits[:, hip_joint_ids_tensor, 1])
    q_knee = q_knee.clamp_(
        joint_pos_limits[:, knee_joint_ids_tensor, 0], joint_pos_limits[:, knee_joint_ids_tensor, 1]
    )
    qd_hip = qd_hip.clamp_(-joint_vel_limits[:, hip_joint_ids_tensor], joint_vel_limits[:, hip_joint_ids_tensor])
    qd_knee = qd_knee.clamp_(-joint_vel_limits[:, knee_joint_ids_tensor], joint_vel_limits[:, knee_joint_ids_tensor])

    asset.write_joint_state_to_sim(q_hip, qd_hip, joint_ids=hip_joint_ids, env_ids=env_ids)
    asset.write_joint_state_to_sim(q_knee, qd_knee, joint_ids=knee_joint_ids, env_ids=env_ids)


def randomize_terrain_physics_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float] = (0.0, 0.0),
    terrain_name: str = "terrain",
    make_consistent: bool = True,
):
    """Randomize terrain physics material (friction/restitution) at startup.

    Note:
        Terrain is typically shared across all environments. Therefore, this randomization is scene-level,
        not per-environment.
    """
    # The event manager always passes env_ids, but terrain material is scene-wide.
    _ = env_ids

    terrain = env.scene[terrain_name]
    terrain_prim_paths = getattr(terrain, "terrain_prim_paths", [])
    if len(terrain_prim_paths) == 0:
        return

    ranges = torch.tensor(
        [static_friction_range, dynamic_friction_range, restitution_range],
        device="cpu",
        dtype=torch.float32,
    )
    material_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(terrain_prim_paths), 3), device="cpu"
    )
    if make_consistent:
        material_samples[:, 1] = torch.minimum(material_samples[:, 0], material_samples[:, 1])

    for idx, terrain_prim_path in enumerate(terrain_prim_paths):
        static_friction = float(material_samples[idx, 0].item())
        dynamic_friction = float(material_samples[idx, 1].item())
        restitution = float(material_samples[idx, 2].item())
        material_path = f"{terrain_prim_path}/physicsMaterial"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        )
        material_cfg.func(material_path, material_cfg)
        sim_utils.bind_physics_material(terrain_prim_path, material_path)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None) # type: ignore
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def bad_orientation_2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot") # type: ignore
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return (asset.data.projected_gravity_b[:, 2] > 0) | (asset.data.projected_gravity_b[:, :2].abs() > 1.4).any(-1)


def illegal_contact_after_time(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    min_time_s: float = 0.0,
) -> torch.Tensor:
    """Terminate on illegal contact only after an initial grace period."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    contact = torch.any(is_contact, dim=1)

    if min_time_s <= 0.0 or not hasattr(env, "episode_length_buf"):
        return contact
    elapsed_s = env.episode_length_buf.to(dtype=torch.float32) * float(env.step_dt)
    return contact & (elapsed_s > float(min_time_s))


def joint_rom_violation(
    env: ManagerBasedEnv,
    q1_upper_limit: float = math.pi,
    q2_lower_limit: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # type: ignore
) -> torch.Tensor:
    """Terminate when any serial q1 exceeds ``q1_upper_limit`` or any serial q2 goes below ``q2_lower_limit``.

    Serial-coordinate convention:
    - q1: HIP
    - q2: KNEE
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = [str(name).upper() for name in asset.joint_names]

    hip_ids = [idx for idx, name in enumerate(joint_names) if "HIP" in name]
    knee_ids = [idx for idx, name in enumerate(joint_names) if "KNEE" in name]
    if len(hip_ids) == 0 or len(knee_ids) == 0:
        raise RuntimeError("joint_rom_violation failed to resolve HIP/KNEE joint ids from asset.joint_names.")

    joint_pos = asset.data.joint_pos
    hip_violation = torch.any(joint_pos[:, hip_ids] > float(q1_upper_limit), dim=1)
    knee_violation = torch.any(joint_pos[:, knee_ids] < float(q2_lower_limit), dim=1)
    return hip_violation | knee_violation
