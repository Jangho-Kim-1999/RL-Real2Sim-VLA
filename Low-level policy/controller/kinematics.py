"""Kinematics helpers used by the controller stack."""

from __future__ import annotations

from typing import Any

import torch

from .dynamics import _resolve_leg_joint_indices, _validate_leg_name


def _resolve_leg_foot_body(asset: Any, leg_name: str, body_name: str | None = None) -> tuple[int, str]:
    leg = _validate_leg_name(leg_name)
    target_body_name = body_name or f"{leg}_foot"

    if not hasattr(asset, "find_bodies"):
        raise AttributeError("Asset does not expose find_bodies(); expected an IsaacLab articulation asset.")

    body_ids, body_names = asset.find_bodies(target_body_name, preserve_order=True)
    if len(body_ids) != 1:
        raise ValueError(
            f"Expected one foot body match for leg {leg}. "
            f"Query='{target_body_name}' resolved to {len(body_ids)} bodies: {body_names}."
        )

    return int(body_ids[0]), str(body_names[0])


def _resolve_jacobian_body_index(asset: Any, body_index: int) -> int:
    if bool(getattr(asset, "is_fixed_base", False)):
        jacobi_body_index = body_index - 1
        if jacobi_body_index < 0:
            raise IndexError(
                f"Jacobian body index became negative for body_index={body_index}. "
                "Check whether the requested body belongs to a fixed-base articulation."
            )
        return jacobi_body_index
    return body_index


def _resolve_jacobian_joint_indices(asset: Any, joint_indices: tuple[int, int, int]) -> tuple[int, int, int]:
    if bool(getattr(asset, "is_fixed_base", False)):
        return joint_indices
    return tuple(index + 6 for index in joint_indices)


def _fetch_jacobians(asset: Any):
    root_physx_view = getattr(asset, "root_physx_view", None)
    if root_physx_view is None or not hasattr(root_physx_view, "get_jacobians"):
        raise AttributeError(
            "Jacobian API not found. Expected asset.root_physx_view.get_jacobians() on the articulation asset."
        )

    jacobians = root_physx_view.get_jacobians()
    if jacobians is None:
        raise RuntimeError("asset.root_physx_view.get_jacobians() returned None.")
    if jacobians.ndim != 4:
        raise ValueError(f"Expected Jacobians with shape [N, B, 6, D], got {tuple(jacobians.shape)}.")
    return jacobians


def _quat_to_rotmat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion in (w, x, y, z) convention to rotation matrix."""
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with last dim 4, got shape {tuple(quat.shape)}.")

    quat = quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1e-12)
    qw, qx, qy, qz = quat.unbind(dim=-1)

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    row0 = torch.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)), dim=-1)
    row1 = torch.stack((2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)), dim=-1)
    row2 = torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _rotate_jacobian_to_body_frame(asset: Any, jacobian_w: torch.Tensor) -> torch.Tensor:
    root_quat_w = getattr(asset.data, "root_quat_w", None)
    if root_quat_w is None:
        raise AttributeError("Asset does not expose asset.data.root_quat_w needed for body-frame Jacobian.")

    if jacobian_w.ndim == 2:
        root_quat_w = root_quat_w[0]
        rot_body_from_world = _quat_to_rotmat_wxyz(root_quat_w).transpose(-1, -2)
        jacobian_b = jacobian_w.clone()
        jacobian_b[:3, :] = rot_body_from_world @ jacobian_w[:3, :]
        jacobian_b[3:, :] = rot_body_from_world @ jacobian_w[3:, :]
        return jacobian_b

    if jacobian_w.ndim != 3:
        raise ValueError(f"Expected Jacobian with ndim 2 or 3, got shape {tuple(jacobian_w.shape)}.")

    rot_body_from_world = _quat_to_rotmat_wxyz(root_quat_w).transpose(-1, -2)
    jacobian_b = jacobian_w.clone()
    jacobian_b[:, :3, :] = torch.bmm(rot_body_from_world, jacobian_w[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(rot_body_from_world, jacobian_w[:, 3:, :])
    return jacobian_b


def get_leg_foot_jacobian_from_sim(
    asset: Any,
    leg_name: str,
    *,
    env_id: int | None = None,
    joint_indices: tuple[int, int, int] | None = None,
    body_name: str | None = None,
):
    """Return the Isaac Sim foot Jacobian for one leg.

    The returned Jacobian corresponds to the requested leg joints ordered as
    ``[HAA, HIP, KNEE]`` and the requested foot body (default: ``"{LEG}_foot"``).

    Returns:
        ``(6, 3)`` if ``env_id`` is provided, otherwise ``(num_envs, 6, 3)``.
    """
    import torch

    joint_ids = _resolve_leg_joint_indices(asset, leg_name, joint_indices)
    body_index, _ = _resolve_leg_foot_body(asset, leg_name, body_name)
    jacobi_body_index = _resolve_jacobian_body_index(asset, body_index)
    jacobi_joint_ids = _resolve_jacobian_joint_indices(asset, joint_ids)

    jacobians = _fetch_jacobians(asset)
    if jacobi_body_index < 0 or jacobi_body_index >= jacobians.shape[1]:
        raise IndexError(
            f"Jacobian body index {jacobi_body_index} out of range for Jacobian tensor with {jacobians.shape[1]} bodies."
        )

    joint_index_tensor = torch.tensor(jacobi_joint_ids, device=jacobians.device, dtype=torch.long)
    leg_jacobian = jacobians[:, jacobi_body_index, :, :].index_select(-1, joint_index_tensor)

    if env_id is not None:
        if env_id < 0 or env_id >= leg_jacobian.shape[0]:
            raise IndexError(f"env_id={env_id} out of range for batch size {leg_jacobian.shape[0]}.")
        return leg_jacobian[env_id]

    return leg_jacobian


def get_leg_foot_linear_jacobian_from_sim(
    asset: Any,
    leg_name: str,
    *,
    env_id: int | None = None,
    joint_indices: tuple[int, int, int] | None = None,
    body_name: str | None = None,
):
    """Return the translational ``3x3`` foot Jacobian for one leg."""
    jacobian = get_leg_foot_jacobian_from_sim(
        asset,
        leg_name,
        env_id=env_id,
        joint_indices=joint_indices,
        body_name=body_name,
    )
    return jacobian[..., :3, :]


def get_leg_foot_jacobian_body_frame_from_sim(
    asset: Any,
    leg_name: str,
    *,
    env_id: int | None = None,
    joint_indices: tuple[int, int, int] | None = None,
    body_name: str | None = None,
):
    """Return the Isaac Sim foot Jacobian expressed in the robot body/base frame."""
    jacobian_w = get_leg_foot_jacobian_from_sim(
        asset,
        leg_name,
        env_id=env_id,
        joint_indices=joint_indices,
        body_name=body_name,
    )
    return _rotate_jacobian_to_body_frame(asset, jacobian_w)


def get_leg_foot_linear_jacobian_body_frame_from_sim(
    asset: Any,
    leg_name: str,
    *,
    env_id: int | None = None,
    joint_indices: tuple[int, int, int] | None = None,
    body_name: str | None = None,
):
    """Return the translational ``3x3`` foot Jacobian expressed in the robot body/base frame."""
    jacobian_b = get_leg_foot_jacobian_body_frame_from_sim(
        asset,
        leg_name,
        env_id=env_id,
        joint_indices=joint_indices,
        body_name=body_name,
    )
    return jacobian_b[..., :3, :]


def get_all_leg_foot_linear_jacobians_from_sim(
    asset: Any,
    *,
    env_id: int | None = None,
) -> dict[str, Any]:
    """Return translational foot Jacobians for all legs."""
    return {
        leg_name: get_leg_foot_linear_jacobian_from_sim(asset, leg_name, env_id=env_id)
        for leg_name in ("FL", "FR", "RL", "RR")
    }


def print_leg_foot_linear_jacobian_from_sim(
    asset: Any,
    leg_name: str,
    *,
    env_id: int = 0,
    precision: int = 10,
) -> None:
    """Print the translational ``3x3`` foot Jacobian for one leg."""
    jacobian = get_leg_foot_linear_jacobian_from_sim(asset, leg_name, env_id=env_id)
    matrix = jacobian.detach().cpu().tolist()
    print(f"[Kinematics] {leg_name.upper()} foot linear Jacobian from Isaac Sim (env_id={env_id}):")
    for row in matrix:
        print(" ".join(f"{float(value): .{precision}e}" for value in row))


def print_leg_foot_linear_jacobian_body_frame_from_sim(
    asset: Any,
    leg_name: str,
    *,
    env_id: int = 0,
    precision: int = 10,
) -> None:
    """Print the translational ``3x3`` foot Jacobian expressed in the robot body/base frame."""
    jacobian = get_leg_foot_linear_jacobian_body_frame_from_sim(asset, leg_name, env_id=env_id)
    matrix = jacobian.detach().cpu().tolist()
    print(f"[Kinematics] {leg_name.upper()} foot linear Jacobian in body frame (env_id={env_id}):")
    for row in matrix:
        print(" ".join(f"{float(value): .{precision}e}" for value in row))


def print_all_leg_foot_linear_jacobians_from_sim(
    asset: Any,
    *,
    env_id: int = 0,
    precision: int = 10,
) -> None:
    """Print the translational ``3x3`` foot Jacobian for all legs."""
    for leg_name in ("FL", "FR", "RL", "RR"):
        print_leg_foot_linear_jacobian_from_sim(asset, leg_name, env_id=env_id, precision=precision)


def print_all_leg_foot_linear_jacobians_body_frame_from_sim(
    asset: Any,
    *,
    env_id: int = 0,
    precision: int = 10,
) -> None:
    """Print the translational ``3x3`` foot Jacobian in the robot body/base frame for all legs."""
    for leg_name in ("FL", "FR", "RL", "RR"):
        print_leg_foot_linear_jacobian_body_frame_from_sim(asset, leg_name, env_id=env_id, precision=precision)
