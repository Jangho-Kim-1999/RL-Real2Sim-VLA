"""Dynamics helpers used by the controller stack."""

from __future__ import annotations

from typing import Any, Final, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

LEG_NAMES: Final[tuple[str, str, str, str]] = ("FL", "FR", "RL", "RR")
LEG_JOINT_SUFFIXES: Final[tuple[str, str, str]] = ("HAA", "HIP", "KNEE")


def _validate_leg_name(leg_name: str) -> str:
    leg = leg_name.upper()
    if leg not in LEG_NAMES:
        raise ValueError(f"Unsupported leg name '{leg_name}'. Expected one of {LEG_NAMES}.")
    return leg


def _get_joint_names(asset: Any) -> list[str]:
    joint_names = getattr(asset, "joint_names", None)
    if joint_names is not None:
        return [str(name) for name in joint_names]

    root_physx_view = getattr(asset, "root_physx_view", None)
    shared_metatype = getattr(root_physx_view, "shared_metatype", None)
    dof_names = getattr(shared_metatype, "dof_names", None)
    if dof_names is not None:
        return [str(name) for name in dof_names]

    return []


def _resolve_leg_joint_indices(
    asset: Any,
    leg_name: str,
    joint_indices: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    if joint_indices is not None:
        if len(joint_indices) != 3:
            raise ValueError(f"joint_indices must have length 3 for {leg_name}; got {joint_indices}.")
        return tuple(int(index) for index in joint_indices)

    leg = _validate_leg_name(leg_name)
    joint_names = _get_joint_names(asset)
    if len(joint_names) == 0:
        raise AttributeError(
            "Joint names not found on asset. Pass joint_indices explicitly or provide an articulation asset."
        )

    expected_joint_names = [f"{leg}{suffix}" for suffix in LEG_JOINT_SUFFIXES]
    missing_joint_names = [joint_name for joint_name in expected_joint_names if joint_name not in joint_names]
    if missing_joint_names:
        raise ValueError(
            f"Failed to resolve {leg} joint names from asset.joint_names. Missing: {missing_joint_names}."
        )

    return tuple(joint_names.index(joint_name) for joint_name in expected_joint_names)


def _fetch_mass_matrix_candidates(asset: Any) -> list[tuple[str, torch.Tensor]]:
    import torch

    generalized_fetch_specs = (
        ("asset.get_generalized_mass_matrices()", asset, "get_generalized_mass_matrices"),
        ("asset.root_physx_view.get_generalized_mass_matrices()", getattr(asset, "root_physx_view", None), "get_generalized_mass_matrices"),
    )
    deprecated_fetch_specs = (
        ("asset.get_mass_matrices()", asset, "get_mass_matrices"),
        ("asset.root_physx_view.get_mass_matrices()", getattr(asset, "root_physx_view", None), "get_mass_matrices"),
    )

    def _read_candidates(fetch_specs: tuple[tuple[str, Any, str], ...]) -> list[tuple[str, torch.Tensor]]:
        candidates: list[tuple[str, torch.Tensor]] = []
        for api_name, owner, attr_name in fetch_specs:
            if owner is None or not hasattr(owner, attr_name):
                continue
            try:
                matrix = getattr(owner, attr_name)()
            except Exception:
                continue
            if torch.is_tensor(matrix) and matrix.ndim >= 2:
                candidates.append((api_name, matrix))
        return candidates

    generalized_candidates = _read_candidates(generalized_fetch_specs)
    if generalized_candidates:
        return generalized_candidates

    return _read_candidates(deprecated_fetch_specs)


def _extract_leg_submatrix(
    matrix: torch.Tensor,
    joint_indices: tuple[int, int, int],
    *,
    env_id: int | None,
) -> torch.Tensor:
    import torch

    num_joints = int(matrix.shape[-1])
    if any(index < 0 or index >= num_joints for index in joint_indices):
        raise IndexError(f"joint_indices {joint_indices} out of range for mass matrix with size {num_joints}.")

    joint_ids = torch.tensor(joint_indices, device=matrix.device, dtype=torch.long)
    if matrix.ndim == 3:
        if env_id is not None:
            if env_id < 0 or env_id >= matrix.shape[0]:
                raise IndexError(f"env_id={env_id} out of range for batch size {matrix.shape[0]}.")
            matrix = matrix[env_id]
        else:
            return matrix.index_select(-2, joint_ids).index_select(-1, joint_ids)
    elif matrix.ndim != 2:
        raise ValueError(f"Expected mass matrix with ndim 2 or 3, got shape {tuple(matrix.shape)}.")

    return matrix.index_select(-2, joint_ids).index_select(-1, joint_ids)


def _resolve_mass_matrix_joint_indices(
    asset: Any,
    matrix: torch.Tensor,
    joint_indices: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Map IsaacLab joint indices to the correct PhysX mass-matrix indices."""
    joint_names = _get_joint_names(asset)
    num_asset_joints = len(joint_names)
    matrix_dim = int(matrix.shape[-1])

    if matrix_dim == num_asset_joints:
        return joint_indices

    is_fixed_base = bool(getattr(asset, "is_fixed_base", False))
    if not is_fixed_base and matrix_dim == num_asset_joints + 6:
        return tuple(index + 6 for index in joint_indices)

    return joint_indices


def _fetch_compensation_vector_candidates(
    asset: Any,
    *,
    kind: str,
) -> list[tuple[str, torch.Tensor]]:
    import torch

    root_physx_view = getattr(asset, "root_physx_view", None)
    if kind == "coriolis":
        fetch_specs = (
            ("asset.get_coriolis_and_centrifugal_compensation_forces()", asset, "get_coriolis_and_centrifugal_compensation_forces"),
            (
                "asset.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()",
                root_physx_view,
                "get_coriolis_and_centrifugal_compensation_forces",
            ),
            ("asset.get_coriolis_and_centrifugal_forces()", asset, "get_coriolis_and_centrifugal_forces"),
            ("asset.root_physx_view.get_coriolis_and_centrifugal_forces()", root_physx_view, "get_coriolis_and_centrifugal_forces"),
        )
    elif kind == "gravity":
        fetch_specs = (
            ("asset.get_gravity_compensation_forces()", asset, "get_gravity_compensation_forces"),
            ("asset.root_physx_view.get_gravity_compensation_forces()", root_physx_view, "get_gravity_compensation_forces"),
            ("asset.get_gravity_forces()", asset, "get_gravity_forces"),
            ("asset.root_physx_view.get_gravity_forces()", root_physx_view, "get_gravity_forces"),
        )
    else:
        raise ValueError(f"Unsupported compensation vector kind '{kind}'.")

    candidates: list[tuple[str, torch.Tensor]] = []
    for api_name, owner, attr_name in fetch_specs:
        if owner is None or not hasattr(owner, attr_name):
            continue
        try:
            vector = getattr(owner, attr_name)()
        except Exception:
            continue
        if torch.is_tensor(vector) and vector.ndim >= 1:
            candidates.append((api_name, vector))
            break

    return candidates


def _resolve_vector_joint_indices(
    asset: Any,
    vector: torch.Tensor,
    joint_indices: tuple[int, int, int],
) -> tuple[int, int, int]:
    joint_names = _get_joint_names(asset)
    num_asset_joints = len(joint_names)
    vector_dim = int(vector.shape[-1])

    if vector_dim == num_asset_joints:
        return joint_indices

    is_fixed_base = bool(getattr(asset, "is_fixed_base", False))
    if not is_fixed_base and vector_dim == num_asset_joints + 6:
        return tuple(index + 6 for index in joint_indices)

    return joint_indices


def _extract_leg_subvector(
    vector: torch.Tensor,
    joint_indices: tuple[int, int, int],
    *,
    env_id: int | None,
) -> torch.Tensor:
    import torch

    num_dofs = int(vector.shape[-1])
    if any(index < 0 or index >= num_dofs for index in joint_indices):
        raise IndexError(f"joint_indices {joint_indices} out of range for vector with size {num_dofs}.")

    joint_ids = torch.tensor(joint_indices, device=vector.device, dtype=torch.long)
    if vector.ndim == 2:
        if env_id is not None:
            if env_id < 0 or env_id >= vector.shape[0]:
                raise IndexError(f"env_id={env_id} out of range for batch size {vector.shape[0]}.")
            vector = vector[env_id]
        else:
            return vector.index_select(-1, joint_ids)
    elif vector.ndim != 1:
        raise ValueError(f"Expected compensation vector with ndim 1 or 2, got shape {tuple(vector.shape)}.")

    return vector.index_select(-1, joint_ids)


def serial_to_parallel_h(h_serial: torch.Tensor) -> torch.Tensor:
    """Map a serial-coordinate generalized force vector to parallel coordinates.

    The coordinate transform uses:

    J = [[1, 0, 0],
         [0, 1, 0],
         [0, 1, 1]]

    and returns:

    h_parallel = J^{-T} h_serial

    for vectors ordered as ``[HAA, HIP, KNEE]``.
    """
    import torch

    if not torch.is_tensor(h_serial):
        raise TypeError("h_serial must be a torch.Tensor.")
    if h_serial.shape[-1] != 3:
        raise ValueError(
            "serial_to_parallel_h expects a trailing dimension of 3 ordered as "
            "[HAA, HIP, KNEE]; "
            f"got shape {tuple(h_serial.shape)}."
        )

    h_parallel = h_serial.clone()
    h_parallel[..., 0] = h_serial[..., 0]
    h_parallel[..., 1] = h_serial[..., 1] - h_serial[..., 2]
    h_parallel[..., 2] = h_serial[..., 2]
    return h_parallel


def _compute_M_parallel_ID_link_serial(
    q2: torch.Tensor | float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_envs: int | None,
) -> torch.Tensor:
    """Return the identified 2x2 serial-link inertia model from q2."""
    import torch

    q2_t = torch.as_tensor(q2, device=device, dtype=dtype)
    if num_envs is not None:
        if q2_t.ndim == 0:
            q2_t = q2_t.expand(num_envs)
        elif q2_t.ndim == 1 and q2_t.shape[0] == num_envs:
            pass
        else:
            raise ValueError(
                f"q2 must be scalar or shape [{num_envs}] when batch output is requested; got {tuple(q2_t.shape)}."
            )
    else:
        if q2_t.ndim > 0:
            if q2_t.numel() != 1:
                raise ValueError(f"q2 must be scalar-like when env_id is set; got shape {tuple(q2_t.shape)}.")
            q2_t = q2_t.reshape(())

    cos_q2 = torch.cos(q2_t)

    m11 = 0.039210619 + 0.010826172 * cos_q2
    m12 = 0.0074822518 + 0.0054130861 * cos_q2
    m22 = torch.full_like(cos_q2, 0.0074822518)

    if num_envs is None:
        return torch.stack(
            [torch.stack([m11, m12]), torch.stack([m12, m22])],
            dim=0,
        )
    return torch.stack(
        [
            torch.stack([m11, m12], dim=-1),
            torch.stack([m12, m22], dim=-1),
        ],
        dim=-2,
    )


def _compute_M_parallel_ID_whole_parallel(
    q2: torch.Tensor | float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_envs: int | None,
) -> torch.Tensor:
    """Return the identified 2x2 whole-parallel inertia model from q2."""
    import torch

    q2_t = torch.as_tensor(q2, device=device, dtype=dtype)
    if num_envs is not None:
        if q2_t.ndim == 0:
            q2_t = q2_t.expand(num_envs)
        elif q2_t.ndim == 1 and q2_t.shape[0] == num_envs:
            pass
        else:
            raise ValueError(
                f"q2 must be scalar or shape [{num_envs}] when batch output is requested; got {tuple(q2_t.shape)}."
            )
    else:
        if q2_t.ndim > 0:
            if q2_t.numel() != 1:
                raise ValueError(f"q2 must be scalar-like when env_id is set; got shape {tuple(q2_t.shape)}.")
            q2_t = q2_t.reshape(())

    cos_q2 = torch.cos(q2_t)

    m11 = torch.full_like(cos_q2, 0.047928367)
    m12 = 0.0054130861 * cos_q2
    m22 = torch.full_like(cos_q2, 0.0236822518)

    if num_envs is None:
        return torch.stack(
            [torch.stack([m11, m12]), torch.stack([m12, m22])],
            dim=0,
        )
    return torch.stack(
        [
            torch.stack([m11, m12], dim=-1),
            torch.stack([m12, m22], dim=-1),
        ],
        dim=-2,
    )


def get_leg_inertia_matrix_from_sim(
    asset: Any,
    leg_name: str,
    *,
    env_id: int | None = None,
    joint_indices: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Return the 3x3 joint-space inertia matrix of one leg from Isaac Sim.

    The returned matrix corresponds to `[HAA, HIP, KNEE]` for the requested leg.
    If `env_id` is `None`, the output shape is `(num_envs, 3, 3)`.
    Otherwise the output shape is `(3, 3)`.
    """
    leg = _validate_leg_name(leg_name)
    matrix_candidates = _fetch_mass_matrix_candidates(asset)
    if len(matrix_candidates) == 0:
        raise AttributeError(
            "Mass matrix API not found. Expected one of: "
            "asset.get_generalized_mass_matrices(), "
            "asset.root_physx_view.get_generalized_mass_matrices(), "
            "asset.get_mass_matrices(), "
            "asset.root_physx_view.get_mass_matrices()."
        )

    resolved_joint_indices = _resolve_leg_joint_indices(asset, leg, joint_indices=joint_indices)

    last_error: Exception | None = None
    for _, matrix in matrix_candidates:
        try:
            matrix_joint_indices = _resolve_mass_matrix_joint_indices(asset, matrix, resolved_joint_indices)
            return _extract_leg_submatrix(matrix, matrix_joint_indices, env_id=env_id)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to extract the {leg} 3x3 inertia matrix from all available Isaac Sim mass-matrix APIs."
    ) from last_error


def get_all_leg_inertia_matrices_from_sim(
    asset: Any,
    *,
    env_id: int | None = None,
) -> dict[str, torch.Tensor]:
    """Return FL/FR/RL/RR 3x3 joint-space inertia matrices from Isaac Sim."""
    return {leg_name: get_leg_inertia_matrix_from_sim(asset, leg_name, env_id=env_id) for leg_name in LEG_NAMES}


def get_fl_inertia_matrix_from_sim(asset: Any, *, env_id: int | None = None) -> torch.Tensor:
    """Return the FL 3x3 joint-space inertia matrix from Isaac Sim."""
    return get_leg_inertia_matrix_from_sim(asset, "FL", env_id=env_id)


def get_fr_inertia_matrix_from_sim(asset: Any, *, env_id: int | None = None) -> torch.Tensor:
    """Return the FR 3x3 joint-space inertia matrix from Isaac Sim."""
    return get_leg_inertia_matrix_from_sim(asset, "FR", env_id=env_id)


def get_rl_inertia_matrix_from_sim(asset: Any, *, env_id: int | None = None) -> torch.Tensor:
    """Return the RL 3x3 joint-space inertia matrix from Isaac Sim."""
    return get_leg_inertia_matrix_from_sim(asset, "RL", env_id=env_id)


def get_rr_inertia_matrix_from_sim(asset: Any, *, env_id: int | None = None) -> torch.Tensor:
    """Return the RR 3x3 joint-space inertia matrix from Isaac Sim."""
    return get_leg_inertia_matrix_from_sim(asset, "RR", env_id=env_id)


def get_leg_C_term_from_sim(
    asset: Any,
    leg_name: str,
    *,
    env_id: int | None = None,
    joint_indices: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Return the 3x1 leg Coriolis/centrifugal term c = C(q, dq) dq from Isaac Sim.

    The returned vector corresponds to `[HAA, HIP, KNEE]` for the requested leg.
    If `env_id` is `None`, the output shape is `(num_envs, 3)`.
    Otherwise the output shape is `(3,)`.
    """
    leg = _validate_leg_name(leg_name)
    vector_candidates = _fetch_compensation_vector_candidates(asset, kind="coriolis")
    if len(vector_candidates) == 0:
        raise AttributeError(
            "Coriolis API not found. Expected one of: "
            "asset.get_coriolis_and_centrifugal_compensation_forces(), "
            "asset.root_physx_view.get_coriolis_and_centrifugal_compensation_forces(), "
            "asset.get_coriolis_and_centrifugal_forces(), "
            "asset.root_physx_view.get_coriolis_and_centrifugal_forces()."
        )

    resolved_joint_indices = _resolve_leg_joint_indices(asset, leg, joint_indices=joint_indices)

    last_error: Exception | None = None
    for _, vector in vector_candidates:
        try:
            vector_joint_indices = _resolve_vector_joint_indices(asset, vector, resolved_joint_indices)
            return _extract_leg_subvector(vector, vector_joint_indices, env_id=env_id)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to extract the {leg} 3x1 Coriolis/centrifugal term from all available Isaac Sim APIs."
    ) from last_error


def get_leg_G_term_from_sim(
    asset: Any,
    leg_name: str,
    *,
    env_id: int | None = None,
    joint_indices: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Return the 3x1 leg gravity term g = G(q) from Isaac Sim.

    The returned vector corresponds to `[HAA, HIP, KNEE]` for the requested leg.
    If `env_id` is `None`, the output shape is `(num_envs, 3)`.
    Otherwise the output shape is `(3,)`.
    """
    leg = _validate_leg_name(leg_name)
    vector_candidates = _fetch_compensation_vector_candidates(asset, kind="gravity")
    if len(vector_candidates) == 0:
        raise AttributeError(
            "Gravity API not found. Expected one of: "
            "asset.get_gravity_compensation_forces(), "
            "asset.root_physx_view.get_gravity_compensation_forces(), "
            "asset.get_gravity_forces(), "
            "asset.root_physx_view.get_gravity_forces()."
        )

    resolved_joint_indices = _resolve_leg_joint_indices(asset, leg, joint_indices=joint_indices)

    last_error: Exception | None = None
    for _, vector in vector_candidates:
        try:
            vector_joint_indices = _resolve_vector_joint_indices(asset, vector, resolved_joint_indices)
            return _extract_leg_subvector(vector, vector_joint_indices, env_id=env_id)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to extract the {leg} 3x1 gravity term from all available Isaac Sim APIs."
    ) from last_error


def get_leg_CG_terms_from_sim(
    asset: Any,
    leg_name: str,
    *,
    env_id: int | None = None,
    joint_indices: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(c_leg, g_leg)` for one leg from Isaac Sim."""
    return (
        get_leg_C_term_from_sim(asset, leg_name, env_id=env_id, joint_indices=joint_indices),
        get_leg_G_term_from_sim(asset, leg_name, env_id=env_id, joint_indices=joint_indices),
    )


def get_leg_C_term_parallel_from_sim(
    asset: Any,
    leg_name: str,
    *,
    env_id: int | None = None,
    joint_indices: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Return the leg Coriolis/centrifugal term in parallel coordinates.

    This computes:
    ``c_parallel = J^{-T} c_serial``
    where ``c_serial`` is the Isaac Sim vector ordered as ``[HAA, HIP, KNEE]``.
    """
    c_serial = get_leg_C_term_from_sim(asset, leg_name, env_id=env_id, joint_indices=joint_indices)
    return serial_to_parallel_h(c_serial)


def get_leg_G_term_parallel_from_sim(
    asset: Any,
    leg_name: str,
    *,
    env_id: int | None = None,
    joint_indices: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Return the leg gravity term in parallel coordinates.

    This computes:
    ``g_parallel = J^{-T} g_serial``
    where ``g_serial`` is the Isaac Sim vector ordered as ``[HAA, HIP, KNEE]``.
    """
    g_serial = get_leg_G_term_from_sim(asset, leg_name, env_id=env_id, joint_indices=joint_indices)
    return serial_to_parallel_h(g_serial)


def get_leg_CG_terms_parallel_from_sim(
    asset: Any,
    leg_name: str,
    *,
    env_id: int | None = None,
    joint_indices: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(c_parallel, g_parallel)`` for one leg from Isaac Sim."""
    c_serial, g_serial = get_leg_CG_terms_from_sim(
        asset,
        leg_name,
        env_id=env_id,
        joint_indices=joint_indices,
    )
    return serial_to_parallel_h(c_serial), serial_to_parallel_h(g_serial)


def get_link_serial_inertia_matrices(
    asset: Any,
    q2: torch.Tensor | float,
    leg_name: str = "FL",
    *,
    env_id: int | None = None,
    joint_indices: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the sim and identified 2x2 hip-knee inertia matrices.

    - M_sim_link_serial: lower-right 2x2 block of the Isaac Sim 3x3 `[HAA, HIP, KNEE]` inertia matrix.
    - M_ID_link_serial: identified 2x2 serial-link inertia model from `q2`.
    """
    M_sim_leg = get_leg_inertia_matrix_from_sim(asset, leg_name, env_id=env_id, joint_indices=joint_indices)
    M_sim_link_serial = M_sim_leg[..., 1:, 1:]

    if M_sim_link_serial.ndim == 3:
        M_ID_link_serial = _compute_M_parallel_ID_link_serial(
            q2,
            device=M_sim_link_serial.device,
            dtype=M_sim_link_serial.dtype,
            num_envs=M_sim_link_serial.shape[0],
        )
    else:
        M_ID_link_serial = _compute_M_parallel_ID_link_serial(
            q2,
            device=M_sim_link_serial.device,
            dtype=M_sim_link_serial.dtype,
            num_envs=None,
        )

    return M_sim_link_serial, M_ID_link_serial


def get_link_parallel_inertia_matrices(
    asset: Any,
    q2: torch.Tensor | float,
    leg_name: str = "FL",
    *,
    env_id: int | None = None,
    joint_indices: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the sim and identified 2x2 parallel inertia matrices.

    - M_sim_link_parallel: lower-right 2x2 block of the Isaac Sim 3x3 `[HAA, HIP, KNEE]` inertia matrix.
    - M_ID_whole_parallel: identified 2x2 whole-parallel inertia model from `q2`.
    """
    M_sim_leg = get_leg_inertia_matrix_from_sim(asset, leg_name, env_id=env_id, joint_indices=joint_indices)
    M_sim_link_parallel = M_sim_leg[..., 1:, 1:]

    if M_sim_link_parallel.ndim == 3:
        M_ID_whole_parallel = _compute_M_parallel_ID_whole_parallel(
            q2,
            device=M_sim_link_parallel.device,
            dtype=M_sim_link_parallel.dtype,
            num_envs=M_sim_link_parallel.shape[0],
        )
    else:
        M_ID_whole_parallel = _compute_M_parallel_ID_whole_parallel(
            q2,
            device=M_sim_link_parallel.device,
            dtype=M_sim_link_parallel.dtype,
            num_envs=None,
        )

    return M_sim_link_parallel, M_ID_whole_parallel


def get_M_ID_whole_parallel(
    q2: torch.Tensor | float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    num_envs: int | None = None,
) -> torch.Tensor:
    """Return the identified 2x2 inertia matrix M_ID_whole_parallel."""
    import torch

    if torch.is_tensor(q2):
        if device is None:
            device = q2.device
        if dtype is None:
            dtype = q2.dtype

    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    return _compute_M_parallel_ID_whole_parallel(
        q2,
        device=device,
        dtype=dtype,
        num_envs=num_envs,
    )


def get_M_3DOF_parallel(
    asset: Any,
    q2: torch.Tensor | float,
    leg_name: str = "FL",
    *,
    env_id: int | None = None,
    joint_indices: tuple[int, int, int] | None = None,
    jm: float = 1.62e-2,
) -> torch.Tensor:
    """Return a mixed 3x3 inertia matrix with sim HAA terms and identified parallel 2x2 block.

    The output uses:
    - the full Isaac Sim 3x3 joint-space inertia matrix `[HAA, HIP, KNEE]` as the base
    - the lower-right 2x2 block replaced by ``M_ID_whole_parallel``
    - the `(1,1)` element incremented by ``jm``

    In other words:
    ``M_3DOF_parallel[..., 1:, 1:] = M_ID_whole_parallel``
    and then ``M_3DOF_parallel[..., 0, 0] += jm``.
    """
    M_sim_leg = get_leg_inertia_matrix_from_sim(asset, leg_name, env_id=env_id, joint_indices=joint_indices)

    if M_sim_leg.ndim == 3:
        M_ID_whole_parallel = get_M_ID_whole_parallel(
            q2,
            device=M_sim_leg.device,
            dtype=M_sim_leg.dtype,
            num_envs=M_sim_leg.shape[0],
        )
    else:
        M_ID_whole_parallel = get_M_ID_whole_parallel(
            q2,
            device=M_sim_leg.device,
            dtype=M_sim_leg.dtype,
            num_envs=None,
        )

    M_3DOF_parallel = M_sim_leg.clone()
    M_3DOF_parallel[..., 1:, 1:] = M_ID_whole_parallel
    M_3DOF_parallel[..., 0, 0] = M_3DOF_parallel[..., 0, 0] + float(jm)
    return M_3DOF_parallel


def get_Delta_M(
    asset: Any,
    q2: torch.Tensor | float,
    leg_name: str = "FL",
    *,
    env_id: int | None = None,
    joint_indices: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Return Delta_M = M_ID_link_serial - M_sim_link_serial."""
    M_sim_link_serial, M_ID_link_serial = get_link_serial_inertia_matrices(
        asset,
        q2,
        leg_name,
        env_id=env_id,
        joint_indices=joint_indices,
    )
    Delta_M = M_ID_link_serial - M_sim_link_serial
    
    return Delta_M


def print_leg_inertia_matrix_from_sim(
    asset: Any,
    leg_name: str = "FL",
    *,
    env_id: int | None = 0,
    joint_indices: tuple[int, int, int] | None = None,
) -> None:
    """Print one leg 3x3 joint-space inertia matrix from Isaac Sim."""
    matrix = get_leg_inertia_matrix_from_sim(asset, leg_name, env_id=env_id, joint_indices=joint_indices)
    leg = _validate_leg_name(leg_name)
    env_label = "all envs" if env_id is None else f"env_id={env_id}"

    print(f"[Dynamics] {leg} joint-space inertia matrix from Isaac Sim ({env_label}):")
    if matrix.ndim == 3:
        for batch_idx, batch_matrix in enumerate(matrix):
            print(f"[Dynamics] {leg} batch[{batch_idx}] (3x3):")
            for row in batch_matrix:
                print("  " + " ".join(f"{float(value): .10e}" for value in row))
    else:
        for row in matrix:
            print("  " + " ".join(f"{float(value): .10e}" for value in row))


def print_all_leg_inertia_matrices_from_sim(asset: Any, *, env_id: int | None = 0) -> None:
    """Print FL/FR/RL/RR 3x3 joint-space inertia matrices from Isaac Sim."""
    for leg_name in LEG_NAMES:
        print_leg_inertia_matrix_from_sim(asset, leg_name, env_id=env_id)
