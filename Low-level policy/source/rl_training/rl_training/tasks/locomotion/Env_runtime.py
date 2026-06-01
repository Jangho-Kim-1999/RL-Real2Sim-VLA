# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import torch
from isaaclab.envs import ManagerBasedRLEnv

try:
    from controller import (
        ForceObserver,
        PDController,
        derivative_filter,
        get_Delta_M,
        get_M_3DOF_parallel,
        get_leg_CG_terms_parallel_from_sim,
        get_leg_foot_linear_jacobian_body_frame_from_sim,
    )
except ModuleNotFoundError:
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.append(_REPO_ROOT)
    from controller import (
        ForceObserver,
        PDController,
        derivative_filter,
        get_Delta_M,
        get_M_3DOF_parallel,
        get_leg_CG_terms_parallel_from_sim,
        get_leg_foot_linear_jacobian_body_frame_from_sim,
    )


class RuntimeTorqueManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based RL env with E2E-style pos->torque application in env step."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self._action_term_name: str = self.cfg.pos_action_term_name
        self._action_term = self.action_manager.get_term(self._action_term_name)
        self._robot = self._action_term._asset
        self._joint_ids = self._action_term._joint_ids
        self._joint_names = self._action_term._joint_names
        self._default_joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids].clone()
        self._q_des = self._default_joint_pos.clone()
        self._last_tau_no_comp = torch.zeros_like(self._q_des)
        self._last_tau_comp = torch.zeros_like(self._q_des)
        self._last_tau_applied = torch.zeros_like(self._q_des)
        self._last_dq_tustin = torch.zeros_like(self._q_des)
        self._last_ddq_tustin = torch.zeros_like(self._q_des)
        self._last_dq_parallel_tustin = torch.zeros_like(self._q_des)
        self._last_ddq_parallel_tustin = torch.zeros_like(self._q_des)
        self._torque_delay_enabled = bool(getattr(self.cfg, "torque_delay_enable", False))
        delay_steps_legacy = max(0, int(getattr(self.cfg, "torque_delay_steps", 0)))
        self._torque_delay_min_steps = max(0, int(getattr(self.cfg, "torque_delay_min_steps", delay_steps_legacy)))
        self._torque_delay_max_steps = max(
            self._torque_delay_min_steps,
            int(getattr(self.cfg, "torque_delay_max_steps", delay_steps_legacy)),
        )
        self._env_delay_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        if self._torque_delay_enabled and self._torque_delay_max_steps > 0:
            self._torque_queue = torch.zeros(
                (self.num_envs, self._torque_delay_max_steps + 1, self._q_des.shape[1]),
                device=self.device,
                dtype=self._q_des.dtype,
            )
        else:
            self._torque_queue = None
        self._pd = PDController(kp=self.cfg.pd_kp, kd=self.cfg.pd_kd)
        self._pd_gain_randomization_enable = bool(getattr(self.cfg, "pd_gain_randomization_enable", False))
        kp_range = getattr(self.cfg, "pd_kp_range", (self._pd.kp, self._pd.kp))
        kd_range = getattr(self.cfg, "pd_kd_range", (self._pd.kd, self._pd.kd))
        self._pd_kp_range = (float(kp_range[0]), float(kp_range[1]))
        self._pd_kd_range = (float(kd_range[0]), float(kd_range[1]))
        if self._pd_kp_range[0] > self._pd_kp_range[1]:
            self._pd_kp_range = (self._pd_kp_range[1], self._pd_kp_range[0])
        if self._pd_kd_range[0] > self._pd_kd_range[1]:
            self._pd_kd_range = (self._pd_kd_range[1], self._pd_kd_range[0])
        self._pd_kp_env = torch.full((self.num_envs, 1), self._pd.kp, device=self.device, dtype=self._q_des.dtype)
        self._pd_kd_env = torch.full((self.num_envs, 1), self._pd.kd, device=self.device, dtype=self._q_des.dtype)
        self.Jm = float(getattr(self.cfg, "Jm", 1.62e-2))
        self.Bm = float(getattr(self.cfg, "Bm", 9.72e-2))
        self._dynamic_conversion_enabled = bool(getattr(self.cfg, "dynamic_conversion_enable", True))
        self._tau_compensation_mode = str(getattr(self.cfg, "tau_compensation_mode", "none")).lower()
        self._pd_control_mode = str(getattr(self.cfg, "pd_control_mode", "effort")).lower()
        if self._pd_control_mode not in ("effort", "implicit"):
            raise ValueError(f"Unsupported pd_control_mode={self._pd_control_mode!r}. Expected effort or implicit.")
        self._hip_knee_pairs = self._resolve_hip_knee_pairs(self.cfg.hip_joint_token, self.cfg.knee_joint_token)
        dt = float(self.physics_dt)
        if dt <= 0.0:
            dt = 1.0e-3
        self._dq_derivative_dt = dt
        self._dq_derivative_cutoff_hz = float(getattr(self.cfg, "dq_derivative_cutoff_hz", 100.0))
        self._ddq_derivative_dt = dt
        self._ddq_derivative_cutoff_hz = float(getattr(self.cfg, "ddq_derivative_cutoff_hz", 30.0))
        self._dq_filter_prev_input = self._default_joint_pos.clone()
        self._dq_filter_prev_output = torch.zeros_like(self._q_des)
        self._ddq_filter_prev_input = torch.zeros_like(self._q_des)
        self._ddq_filter_prev_output = torch.zeros_like(self._q_des)
        self._play_logging_enabled = False
        self._play_log_env_id = 0
        self._play_log_buffer: list[dict[str, object]] = []
        self._play_contact_threshold = 0.0
        self._play_contact_names, self._play_contact_body_ids = self._resolve_play_contact_bodies()
        self._grf_body_names, self._grf_body_ids = self._resolve_grf_bodies()
        self._grf_robot_body_ids = self._resolve_robot_body_ids(self._grf_body_names)
        self._force_observer = ForceObserver()
        self._leg_joint_triplets = self._resolve_leg_joint_triplets()

    def _resolve_leg_joint_triplets(self) -> dict[str, tuple[int, int, int]]:
        triplets: dict[str, tuple[int, int, int]] = {}
        for leg in ("FL", "FR", "RL", "RR"):
            idx_haa = None
            idx_hip = None
            idx_knee = None
            for joint_idx, joint_name in enumerate(self._joint_names):
                name = str(joint_name).upper()
                if not name.startswith(leg):
                    continue
                if "HAA" in name:
                    idx_haa = joint_idx
                elif "HIP" in name:
                    idx_hip = joint_idx
                elif "KNEE" in name:
                    idx_knee = joint_idx
            if idx_haa is not None and idx_hip is not None and idx_knee is not None:
                triplets[leg] = (idx_haa, idx_hip, idx_knee)
        return triplets

    def _compute_play_force_observer(self, env_index: int) -> tuple[list[str], torch.Tensor | None, torch.Tensor | None]:
        if not self._leg_joint_triplets:
            return [], None, None

        q = self._robot.data.joint_pos[env_index, self._joint_ids]
        tau_no_comp = self._last_tau_no_comp[env_index]
        dq_parallel = self._last_dq_parallel_tustin[env_index]
        ddq_parallel = self._last_ddq_parallel_tustin[env_index]

        leg_names: list[str] = []
        residual_rows: list[torch.Tensor] = []
        force_rows: list[torch.Tensor] = []
        for leg, (i_haa, i_hip, i_knee) in self._leg_joint_triplets.items():
            try:
                tau_leg = torch.stack([tau_no_comp[i_haa], tau_no_comp[i_hip], tau_no_comp[i_knee]], dim=0).unsqueeze(0)
                q_leg = torch.stack([q[i_haa], q[i_hip], q[i_hip] + q[i_knee]], dim=0).unsqueeze(0)
                dq_leg = torch.stack([dq_parallel[i_haa], dq_parallel[i_hip], dq_parallel[i_knee]], dim=0).unsqueeze(0)
                ddq_leg = torch.stack([ddq_parallel[i_haa], ddq_parallel[i_hip], ddq_parallel[i_knee]], dim=0).unsqueeze(0)
                c_parallel, g_parallel = get_leg_CG_terms_parallel_from_sim(self._robot, leg, env_id=env_index)
                h_parallel = (c_parallel + g_parallel).unsqueeze(0)
                m_parallel = get_M_3DOF_parallel(
                    self._robot,
                    q[i_knee].unsqueeze(0),
                    leg_name=leg,
                    env_id=env_index,
                    jm=self.Jm,
                ).unsqueeze(0)
                residual = self._force_observer(
                    tau_input=tau_leg,
                    q_input=q_leg,
                    h_parallel=h_parallel,
                    M_parallel=m_parallel,
                    dq_parallel_tustin=dq_leg,
                    ddq_parallel_tustin=ddq_leg,
                    motor_damping=self.Bm,
                )["residual"][0]
                jacobian_b = get_leg_foot_linear_jacobian_body_frame_from_sim(self._robot, leg, env_id=env_index)
                force_body = torch.linalg.pinv(jacobian_b.transpose(-1, -2), rcond=1.0e-4) @ residual

                leg_names.append(leg)
                residual_rows.append(residual)
                force_rows.append(force_body)
            except Exception:
                continue

        if not leg_names:
            return [], None, None
        return leg_names, torch.stack(residual_rows, dim=0), torch.stack(force_rows, dim=0)

    def _resolve_hip_knee_pairs(self, hip_token: str, knee_token: str) -> list[tuple[int, int]]:
        hip_token_upper = hip_token.upper()
        knee_token_upper = knee_token.upper()

        knee_by_prefix: dict[str, int] = {}
        for idx, name in enumerate(self._joint_names):
            name_upper = name.upper()
            knee_pos = name_upper.find(knee_token_upper)
            if knee_pos >= 0:
                knee_by_prefix[name[:knee_pos]] = idx

        pairs: list[tuple[int, int]] = []
        for idx, name in enumerate(self._joint_names):
            name_upper = name.upper()
            hip_pos = name_upper.find(hip_token_upper)
            if hip_pos < 0:
                continue
            prefix = name[:hip_pos]
            knee_idx = knee_by_prefix.get(prefix)
            if knee_idx is not None:
                pairs.append((idx, knee_idx))

        pairs.sort(key=lambda pair: pair[0])
        return pairs

    def _sample_pd_gains(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if self._pd_gain_randomization_enable:
            kp = torch.empty(env_ids.shape[0], device=self.device, dtype=self._q_des.dtype)
            kd = torch.empty(env_ids.shape[0], device=self.device, dtype=self._q_des.dtype)
            kp.uniform_(*self._pd_kp_range)
            kd.uniform_(*self._pd_kd_range)
            self._pd_kp_env[env_ids, 0] = kp
            self._pd_kd_env[env_ids, 0] = kd
        else:
            self._pd_kp_env[env_ids, 0] = self._pd.kp
            self._pd_kd_env[env_ids, 0] = self._pd.kd

    def _compute_pd_torque(self, position_error: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        kp = self._pd_kp_env
        kd = self._pd_kd_env
        while kp.ndim < position_error.ndim:
            kp = kp.unsqueeze(-1)
            kd = kd.unsqueeze(-1)
        return kp * position_error - kd * velocity

    def _compute_serial_target_from_bispace_action(self, q_des: torch.Tensor) -> torch.Tensor:
        q_target = q_des.clone()
        if self.cfg.use_biarticular_hip_knee and len(self._hip_knee_pairs) > 0:
            hip_ids = [pair[0] for pair in self._hip_knee_pairs]
            knee_ids = [pair[1] for pair in self._hip_knee_pairs]
            action_delta = q_des - self._default_joint_pos
            qm_des = self._default_joint_pos[:, hip_ids] + action_delta[:, hip_ids]
            qb_des = self._default_joint_pos[:, hip_ids] + self._default_joint_pos[:, knee_ids] + action_delta[:, knee_ids]
            q_target[:, hip_ids] = qm_des
            q_target[:, knee_ids] = qb_des - qm_des
        return q_target

    def _update_implicit_pd_torque_logs(self, q_target: torch.Tensor, q: torch.Tensor, dq: torch.Tensor) -> None:
        tau_est = self._compute_pd_torque(q_target - q, dq)
        if self.cfg.torque_limit is not None:
            tau_est = torch.clamp(tau_est, -self.cfg.torque_limit, self.cfg.torque_limit)
        if self.cfg.use_biarticular_hip_knee and len(self._hip_knee_pairs) > 0:
            hip_ids = [pair[0] for pair in self._hip_knee_pairs]
            knee_ids = [pair[1] for pair in self._hip_knee_pairs]
            tau_no_comp_log = tau_est.clone()
            tau_no_comp_log[:, hip_ids] = tau_est[:, hip_ids] - tau_est[:, knee_ids]
            tau_no_comp_log[:, knee_ids] = tau_est[:, knee_ids]
        else:
            tau_no_comp_log = tau_est
        self._last_tau_no_comp = tau_no_comp_log
        self._last_tau_comp = torch.zeros_like(tau_est)
        self._last_tau_applied = tau_est.clone()

    def _estimate_tustin_joint_dynamics(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dq_tustin = derivative_filter(
            q,
            self._dq_filter_prev_input,
            self._dq_filter_prev_output,
            dt=self._dq_derivative_dt,
            cutoff_hz=self._dq_derivative_cutoff_hz,
        )
        ddq_tustin = derivative_filter(
            dq_tustin,
            self._ddq_filter_prev_input,
            self._ddq_filter_prev_output,
            dt=self._ddq_derivative_dt,
            cutoff_hz=self._ddq_derivative_cutoff_hz,
        )
        self._dq_filter_prev_input = q.clone()
        self._dq_filter_prev_output = dq_tustin.clone()
        self._ddq_filter_prev_input = dq_tustin.clone()
        self._ddq_filter_prev_output = ddq_tustin.clone()
        return dq_tustin, ddq_tustin

    def _serial_to_parallel_state(self, serial_state: torch.Tensor) -> torch.Tensor:
        """Map serial joint state [HAA, HIP, KNEE] to parallel coordinates [q0, qm, qb].

        The tensor keeps the same joint-slot layout:
        - HAA slot stays serial.
        - HIP slot stores qm.
        - KNEE slot stores qb = qhip + qknee.
        """
        parallel_state = serial_state.clone()
        if self.cfg.use_biarticular_hip_knee and len(self._hip_knee_pairs) > 0:
            hip_ids = [pair[0] for pair in self._hip_knee_pairs]
            knee_ids = [pair[1] for pair in self._hip_knee_pairs]
            parallel_state[:, hip_ids] = serial_state[:, hip_ids]
            parallel_state[:, knee_ids] = serial_state[:, hip_ids] + serial_state[:, knee_ids]
        return parallel_state

    def _resolve_play_contact_bodies(self) -> tuple[list[str], list[int]]:
        try:
            contact_sensor = self.scene.sensors["contact_forces"]
        except Exception:
            return [], []

        foot_names_cfg = []
        if hasattr(self.cfg, "whole_link_names"):
            foot_names_cfg = [str(name) for name in getattr(self.cfg, "whole_link_names", []) if "foot" in str(name).lower()]
        if not foot_names_cfg and hasattr(self.cfg, "link_names"):
            foot_names_cfg = [str(name) for name in getattr(self.cfg, "link_names", []) if "foot" in str(name).lower()]
        if foot_names_cfg:
            body_ids, body_names = contact_sensor.find_bodies(foot_names_cfg, preserve_order=True)
        else:
            foot_expr = str(getattr(self.cfg, "foot_link_name", ".*_foot"))
            body_ids, body_names = contact_sensor.find_bodies([foot_expr], preserve_order=True)
        return [str(name) for name in body_names], [int(idx) for idx in body_ids]

    def _get_grf_sensor(self):
        try:
            return "ground_reaction_forces", self.scene.sensors["ground_reaction_forces"]
        except Exception:
            return "contact_forces", self.scene.sensors["contact_forces"]

    def _resolve_grf_bodies(self) -> tuple[list[str], list[int]]:
        try:
            _, grf_sensor = self._get_grf_sensor()
        except Exception:
            return [], []

        foot_names_cfg = []
        if hasattr(self.cfg, "whole_link_names"):
            foot_names_cfg = [str(name) for name in getattr(self.cfg, "whole_link_names", []) if "foot" in str(name).lower()]
        if not foot_names_cfg and hasattr(self.cfg, "link_names"):
            foot_names_cfg = [str(name) for name in getattr(self.cfg, "link_names", []) if "foot" in str(name).lower()]
        if foot_names_cfg:
            body_ids, body_names = grf_sensor.find_bodies(foot_names_cfg, preserve_order=True)
        else:
            foot_expr = str(getattr(self.cfg, "foot_link_name", ".*_foot"))
            body_ids, body_names = grf_sensor.find_bodies([foot_expr], preserve_order=True)
        return [str(name) for name in body_names], [int(idx) for idx in body_ids]

    def _resolve_robot_body_ids(self, body_names: list[str]) -> list[int]:
        if not body_names:
            return []
        try:
            body_ids, _ = self._robot.find_bodies(body_names, preserve_order=True)
            if len(body_ids) == len(body_names):
                return [int(idx) for idx in body_ids]
        except Exception:
            pass
        return []

    def _select_env_rows(self, tensor: torch.Tensor, env_ids: int | list[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return tensor
        if isinstance(env_ids, int):
            return tensor[env_ids]
        if torch.is_tensor(env_ids):
            env_ids = env_ids.to(device=tensor.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=tensor.device, dtype=torch.long)
        return tensor[env_ids]

    def _get_grf_body_origins_w(self, num_bodies: int, dtype: torch.dtype) -> torch.Tensor | None:
        if len(self._grf_robot_body_ids) != int(num_bodies):
            return None
        for attr in ("body_link_pos_w", "body_pos_w"):
            body_pos_w = getattr(self._robot.data, attr, None)
            if body_pos_w is None:
                continue
            try:
                return body_pos_w[:, self._grf_robot_body_ids, :].to(device=self.device, dtype=dtype)
            except Exception:
                continue
        return None

    def _as_device_tensor(self, value, dtype: torch.dtype | None = None) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value.to(device=self.device)
        else:
            tensor = torch.as_tensor(value, device=self.device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _aggregate_contact_stream_w(
        self,
        forces: torch.Tensor,
        points: torch.Tensor,
        pair_counts: torch.Tensor,
        pair_start_indices: torch.Tensor,
        origins_w: torch.Tensor | None,
        *,
        normal_vectors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_counts = pair_counts.to(device=self.device, dtype=torch.long)
        pair_start_indices = pair_start_indices.to(device=self.device, dtype=torch.long)
        forces = forces.to(device=self.device)
        points = points.to(device=self.device, dtype=forces.dtype)
        if normal_vectors is not None:
            normal_vectors = normal_vectors.to(device=self.device, dtype=forces.dtype)

        num_envs = int(self.num_envs)
        num_pairs = int(pair_counts.shape[0])
        try:
            num_bodies = int(num_pairs // max(num_envs, 1))
        except Exception:
            num_bodies = 0
        if num_envs <= 0 or num_bodies <= 0 or num_pairs != num_envs * num_bodies:
            empty = torch.zeros((num_envs, 0, 3), device=self.device, dtype=forces.dtype)
            return empty, empty

        counts_flat = pair_counts.reshape(num_pairs, -1)
        starts_flat = pair_start_indices.reshape(num_pairs, -1)
        force_flat = torch.zeros((num_pairs, 3), device=self.device, dtype=forces.dtype)
        moment_flat = torch.zeros_like(force_flat)
        max_contact_count = int(forces.shape[0])
        if origins_w is not None and origins_w.shape[:2] == (num_envs, num_bodies):
            origins_flat = origins_w.reshape(num_pairs, 3).to(device=self.device, dtype=forces.dtype)
        else:
            origins_flat = torch.zeros((num_pairs, 3), device=self.device, dtype=forces.dtype)

        for pair_idx in range(num_pairs):
            origin = origins_flat[pair_idx]
            for filter_idx in range(counts_flat.shape[1]):
                count = int(counts_flat[pair_idx, filter_idx].item())
                start = int(starts_flat[pair_idx, filter_idx].item())
                if count <= 0 or start < 0 or start >= max_contact_count:
                    continue
                end = min(start + count, max_contact_count)
                if end <= start:
                    continue
                if normal_vectors is None:
                    force_vec = forces[start:end]
                elif forces.shape[-1] == 1:
                    force_vec = forces[start:end].reshape(-1, 1) * normal_vectors[start:end]
                else:
                    force_vec = forces[start:end]
                contact_points = points[start:end]
                force_flat[pair_idx] += force_vec.sum(dim=0)
                moment_flat[pair_idx] += torch.cross(contact_points - origin, force_vec, dim=-1).sum(dim=0)

        shape = (num_envs, num_bodies, 3)
        return (
            torch.nan_to_num(force_flat.view(shape), nan=0.0),
            torch.nan_to_num(moment_flat.view(shape), nan=0.0),
        )

    def _get_filtered_contact_wrench_w(self, contact_sensor) -> dict[str, torch.Tensor] | None:
        """Aggregate filtered normal/friction forces and moments about foot origins."""
        contact_view = getattr(contact_sensor, "contact_physx_view", None)
        if contact_view is None:
            return None

        normal_force = None
        normal_moment = None
        friction_force = None
        friction_moment = None
        reference_dtype = torch.float32

        contact_data = None
        try:
            if hasattr(contact_view, "get_contact_data"):
                contact_data = contact_view.get_contact_data(dt=float(self.physics_dt))
            elif hasattr(contact_view, "get_contact_force_data"):
                contact_data = contact_view.get_contact_force_data(dt=float(self.physics_dt))
        except Exception:
            contact_data = None
        if contact_data is not None and len(contact_data) >= 6:
            normal_values, normal_points, normal_vectors, _distances, pair_counts, pair_starts = contact_data[:6]
            if normal_values is not None and normal_points is not None and pair_counts is not None and pair_starts is not None:
                normal_values = self._as_device_tensor(normal_values)
                reference_dtype = normal_values.dtype
                origins_w = self._get_grf_body_origins_w(int(pair_counts.shape[0]) // max(int(self.num_envs), 1), reference_dtype)
                normal_force, normal_moment = self._aggregate_contact_stream_w(
                    normal_values,
                    self._as_device_tensor(normal_points, dtype=reference_dtype),
                    self._as_device_tensor(pair_counts, dtype=torch.long),
                    self._as_device_tensor(pair_starts, dtype=torch.long),
                    origins_w,
                    normal_vectors=self._as_device_tensor(normal_vectors, dtype=reference_dtype)
                    if normal_vectors is not None
                    else None,
                )

        friction_data = None
        try:
            if hasattr(contact_view, "get_friction_data"):
                friction_data = contact_view.get_friction_data(dt=float(self.physics_dt))
        except Exception:
            friction_data = None
        if friction_data is not None and len(friction_data) >= 4:
            friction_values, friction_points, pair_counts, pair_starts = friction_data[:4]
            if friction_values is not None and friction_points is not None and pair_counts is not None and pair_starts is not None:
                friction_values = self._as_device_tensor(friction_values)
                reference_dtype = friction_values.dtype
                origins_w = self._get_grf_body_origins_w(int(pair_counts.shape[0]) // max(int(self.num_envs), 1), reference_dtype)
                friction_force, friction_moment = self._aggregate_contact_stream_w(
                    friction_values,
                    self._as_device_tensor(friction_points, dtype=reference_dtype),
                    self._as_device_tensor(pair_counts, dtype=torch.long),
                    self._as_device_tensor(pair_starts, dtype=torch.long),
                    origins_w,
                )

        template = normal_force if normal_force is not None else friction_force
        if template is None:
            return None
        if normal_force is None:
            normal_force = torch.zeros_like(template)
            normal_moment = torch.zeros_like(template)
        if friction_force is None:
            friction_force = torch.zeros_like(template)
            friction_moment = torch.zeros_like(template)
        return {
            "normal_force": normal_force,
            "friction_force": friction_force,
            "total_force": normal_force + friction_force,
            "normal_moment": normal_moment,
            "friction_moment": friction_moment,
            "total_moment": normal_moment + friction_moment,
        }

    def _get_filtered_contact_friction_forces_w(self, contact_sensor) -> torch.Tensor | None:
        wrench = self._get_filtered_contact_wrench_w(contact_sensor)
        if wrench is None:
            return None
        return wrench["friction_force"]

    def get_ground_reaction_force_components(
        self, env_ids: int | list[int] | torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | list[str]]:
        sensor_name, grf_sensor = self._get_grf_sensor()
        if not self._grf_body_ids:
            raise RuntimeError("No foot bodies were resolved for ground reaction force sensing.")

        if sensor_name == "ground_reaction_forces" and grf_sensor.data.force_matrix_w is not None:
            normal = torch.nan_to_num(grf_sensor.data.force_matrix_w[:, self._grf_body_ids], nan=0.0).sum(dim=2)
        else:
            if grf_sensor.data.net_forces_w is None:
                raise RuntimeError("Contact sensor does not expose net contact forces.")
            normal = torch.nan_to_num(grf_sensor.data.net_forces_w[:, self._grf_body_ids], nan=0.0)

        wrench = self._get_filtered_contact_wrench_w(grf_sensor)
        normal_moment = None
        friction_moment = None
        friction = None
        friction_data = getattr(grf_sensor.data, "friction_forces_w", None)
        if friction_data is not None:
            friction = torch.nan_to_num(friction_data[:, self._grf_body_ids], nan=0.0).sum(dim=2)
        else:
            if wrench is not None:
                friction = wrench["friction_force"][:, self._grf_body_ids]
        if friction is None:
            friction = torch.zeros_like(normal)
        if wrench is not None:
            normal_moment = wrench["normal_moment"][:, self._grf_body_ids]
            friction_moment = wrench["friction_moment"][:, self._grf_body_ids]
        if normal_moment is None:
            normal_moment = torch.zeros_like(normal)
        if friction_moment is None:
            friction_moment = torch.zeros_like(normal)

        total = normal + friction
        total_moment = normal_moment + friction_moment
        return {
            "foot_names": list(self._grf_body_names),
            "total": self._select_env_rows(total, env_ids),
            "normal": self._select_env_rows(normal, env_ids),
            "friction": self._select_env_rows(friction, env_ids),
            "moment": self._select_env_rows(total_moment, env_ids),
            "normal_moment": self._select_env_rows(normal_moment, env_ids),
            "friction_moment": self._select_env_rows(friction_moment, env_ids),
        }

    def get_ground_reaction_forces(self, env_ids: int | list[int] | torch.Tensor | None = None) -> torch.Tensor:
        return self.get_ground_reaction_force_components(env_ids)["total"]  # type: ignore[return-value]

    def get_ground_reaction_force_norms(self, env_ids: int | list[int] | torch.Tensor | None = None) -> torch.Tensor:
        return torch.linalg.vector_norm(self.get_ground_reaction_forces(env_ids), dim=-1)

    def _quat_wxyz_to_rpy(self, quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = quat_wxyz[:, 0]
        x = quat_wxyz[:, 1]
        y = quat_wxyz[:, 2]
        z = quat_wxyz[:, 3]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def _quat_rotate_inverse(self, quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate world-frame vectors into the quaternion's local/body frame."""
        q_w = quat_wxyz[..., 0:1]
        q_xyz = quat_wxyz[..., 1:4]
        return (
            vec * (2.0 * q_w * q_w - 1.0)
            - 2.0 * q_w * torch.cross(q_xyz, vec, dim=-1)
            + 2.0 * q_xyz * torch.sum(q_xyz * vec, dim=-1, keepdim=True)
        )

    def configure_play_logging(self, env_id: int = 0, enable: bool = True) -> None:
        self._play_logging_enabled = bool(enable)
        self._play_log_env_id = min(max(int(env_id), 0), self.num_envs - 1)
        self._play_log_buffer.clear()

    def drain_play_log_buffer(self) -> list[dict[str, object]]:
        buffered = self._play_log_buffer
        self._play_log_buffer = []
        return buffered

    def _get_play_contact_state(self, env_index: int) -> list[int]:
        if not self._play_contact_body_ids:
            return []
        try:
            contact_sensor = self.scene.sensors["contact_forces"]
        except Exception:
            return []
        net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, self._play_contact_body_ids]
        contact_now = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > self._play_contact_threshold
        return [int(value) for value in contact_now[env_index].to(dtype=torch.int32).detach().cpu().tolist()]

    def _record_play_log_sample(self, sim_step: int) -> None:
        if not self._play_logging_enabled:
            return
        env_index = self._play_log_env_id
        q = self._robot.data.joint_pos[env_index, self._joint_ids].detach().cpu().clone()
        dq_raw = self._robot.data.joint_vel[env_index, self._joint_ids].detach().cpu().clone()
        ddq_raw = self._robot.data.joint_acc[env_index, self._joint_ids].detach().cpu().clone()
        q_des = self._q_des[env_index].detach().cpu().clone()
        dq_tustin = self._last_dq_tustin[env_index].detach().cpu().clone()
        ddq_tustin = self._last_ddq_tustin[env_index].detach().cpu().clone()
        dq_parallel_tustin = self._last_dq_parallel_tustin[env_index].detach().cpu().clone()
        ddq_parallel_tustin = self._last_ddq_parallel_tustin[env_index].detach().cpu().clone()
        tau_no_comp = self._last_tau_no_comp[env_index].detach().cpu().clone()
        tau_comp = self._last_tau_comp[env_index].detach().cpu().clone()
        applied_torque = self._robot.data.applied_torque[env_index, self._joint_ids].detach().cpu().clone()
        base_lin_vel = self._robot.data.root_lin_vel_b[env_index, :3].detach().cpu().clone()
        base_ang_vel = self._robot.data.root_ang_vel_b[env_index, :3].detach().cpu().clone()
        base_quat = self._robot.data.root_quat_w[env_index : env_index + 1].detach().cpu().clone()
        base_pos = self._robot.data.root_pos_w[env_index, :3].detach().cpu().clone()
        roll, pitch, yaw = self._quat_wxyz_to_rpy(base_quat)
        base_speed_xy = torch.linalg.vector_norm(base_lin_vel[:2], ord=2).item()
        base_tilt = torch.sqrt(roll[0] ** 2 + pitch[0] ** 2).item()
        base_command = None
        grf_names: list[str] = []
        grf_total = None
        grf_normal = None
        grf_friction = None
        grf_moment = None
        grf_normal_moment = None
        grf_friction_moment = None
        fob_leg_names: list[str] = []
        fob_residual = None
        fob_force_body = None
        sensor_force_body = None
        max_contact_body_name = ""
        max_contact_force_norm = float("nan")
        if hasattr(self, "command_manager"):
            for command_name in ("base_velocity", "body_pose", "jump_mode"):
                try:
                    maybe_cmd = self.command_manager.get_command(command_name)
                    if torch.is_tensor(maybe_cmd):
                        base_command = maybe_cmd[env_index].detach().cpu().clone()
                        break
                except Exception:
                    continue
        # Prefer ground_reaction_forces for play force logging because it is
        # configured on the foot bodies and exposes x/y/z contact-force vectors.
        try:
            grf_components = self.get_ground_reaction_force_components(env_index)
            grf_names = [str(name) for name in grf_components["foot_names"]]
            grf_total = grf_components["total"].detach().cpu().clone()
            grf_normal = grf_components["normal"].detach().cpu().clone()
            grf_friction = grf_components["friction"].detach().cpu().clone()
            grf_moment = grf_components["moment"].detach().cpu().clone()
            grf_normal_moment = grf_components["normal_moment"].detach().cpu().clone()
            grf_friction_moment = grf_components["friction_moment"].detach().cpu().clone()
        except Exception:
            pass

        # Keep contact_forces for contact debug and as a fallback if the
        # dedicated GRF sensor is unavailable.
        try:
            contact_sensor = self.scene.sensors["contact_forces"]
            body_ids = self._play_contact_body_ids
            body_names = self._play_contact_names
            all_force_history = getattr(contact_sensor.data, "net_forces_w_history", None)
            if all_force_history is not None:
                all_forces = all_force_history[env_index]
                all_force_norms = torch.linalg.vector_norm(all_forces, dim=-1)
                flat_max_idx = torch.argmax(all_force_norms.reshape(-1))
                body_count = all_force_norms.shape[-1]
                body_idx = int((flat_max_idx % body_count).detach().cpu().item())
                max_contact_force_norm = float(all_force_norms.reshape(-1)[flat_max_idx].detach().cpu().item())
                try:
                    max_contact_body_name = str(contact_sensor.body_names[body_idx])
                except Exception:
                    max_contact_body_name = str(body_idx)
            if (
                grf_total is None
                and len(body_ids) > 0
                and getattr(contact_sensor.data, "net_forces_w_history", None) is not None
            ):
                force_history = contact_sensor.data.net_forces_w_history[env_index, :, body_ids, :]
                force_norm_history = torch.linalg.vector_norm(force_history, dim=-1)
                max_history_ids = torch.argmax(force_norm_history, dim=0)
                foot_ids = torch.arange(len(body_ids), device=force_history.device)
                forces = force_history[max_history_ids, foot_ids]
                grf_total = torch.nan_to_num(forces, nan=0.0).detach().cpu().clone()
                grf_normal = grf_total.clone()
                grf_friction = torch.zeros_like(grf_total)
                grf_moment = torch.zeros_like(grf_total)
                grf_normal_moment = torch.zeros_like(grf_total)
                grf_friction_moment = torch.zeros_like(grf_total)
                grf_names = [str(name) for name in body_names]
            elif grf_total is None and len(body_ids) > 0 and contact_sensor.data.net_forces_w is not None:
                forces = contact_sensor.data.net_forces_w[env_index, body_ids]
                grf_total = torch.nan_to_num(forces, nan=0.0).detach().cpu().clone()
                grf_normal = grf_total.clone()
                grf_friction = torch.zeros_like(grf_total)
                grf_moment = torch.zeros_like(grf_total)
                grf_normal_moment = torch.zeros_like(grf_total)
                grf_friction_moment = torch.zeros_like(grf_total)
                grf_names = [str(name) for name in body_names]
        except Exception:
            pass
        # Fallback to GRF helper if contact_forces path is unavailable.
        if grf_total is None:
            try:
                grf_components = self.get_ground_reaction_force_components(env_index)
                grf_names = [str(name) for name in grf_components["foot_names"]]
                grf_total = grf_components["total"].detach().cpu().clone()
                grf_normal = grf_components["normal"].detach().cpu().clone()
                grf_friction = grf_components["friction"].detach().cpu().clone()
                grf_moment = grf_components["moment"].detach().cpu().clone()
                grf_normal_moment = grf_components["normal_moment"].detach().cpu().clone()
                grf_friction_moment = grf_components["friction_moment"].detach().cpu().clone()
            except Exception:
                pass
        try:
            fob_leg_names, fob_residual, fob_force_body = self._compute_play_force_observer(env_index)
        except Exception:
            pass
        if grf_total is not None:
            try:
                body_quat = base_quat.expand(int(grf_total.shape[0]), -1)
                sensor_force_body = self._quat_rotate_inverse(body_quat, grf_total)
            except Exception:
                sensor_force_body = None

        self._play_log_buffer.append(
            {
                "sim_step": int(sim_step),
                "time_s": float(sim_step) * float(self.physics_dt),
                "env_index": int(env_index),
                "joint_names": list(self._joint_names),
                "contact_names": list(self._play_contact_names),
                "joint_pos": q,
                "joint_vel": dq_raw,
                "joint_acc": ddq_raw,
                "joint_vel_raw": dq_raw,
                "joint_acc_raw": ddq_raw,
                "joint_vel_tustin": dq_tustin,
                "joint_acc_tustin": ddq_tustin,
                "joint_vel_parallel_tustin": dq_parallel_tustin,
                "joint_acc_parallel_tustin": ddq_parallel_tustin,
                "target_joint_pos": q_des,
                "tau_no_comp": tau_no_comp,
                "tau_comp": tau_comp,
                "applied_torque": applied_torque,
                "base_lin_vel": base_lin_vel,
                "base_ang_vel": base_ang_vel,
                "command": base_command,
                "base_speed_xy": base_speed_xy,
                "base_roll": roll[0].item(),
                "base_pitch": pitch[0].item(),
                "base_yaw": yaw[0].item(),
                "base_tilt": base_tilt,
                "base_pos_x": base_pos[0].item(),
                "base_pos_y": base_pos[1].item(),
                "base_pos_z": base_pos[2].item(),
                "base_height": base_pos[2].item(),
                "contact_state": self._get_play_contact_state(env_index),
                "grf_names": grf_names,
                "W_grf": grf_total,
                "W_grf_normal": grf_normal,
                "W_grf_friction": grf_friction,
                "W_grf_moment": grf_moment,
                "W_grf_normal_moment": grf_normal_moment,
                "W_grf_friction_moment": grf_friction_moment,
                "fob_leg_names": fob_leg_names,
                "fob_residual": fob_residual,
                "fob_force_body": fob_force_body,
                "sensor_force_body": sensor_force_body,
                "max_contact_body_name": max_contact_body_name,
                "max_contact_force_norm": max_contact_force_norm,
            }
        )

    def _pre_physics_step(self, action: torch.Tensor) -> None:
        self.action_manager.process_action(action.to(self.device))
        self._q_des = self._action_term.processed_actions

    def _apply_action(self) -> None:
        q = self._robot.data.joint_pos[:, self._joint_ids]
        dq_raw = self._robot.data.joint_vel[:, self._joint_ids]
        ddq_raw = self._robot.data.joint_acc[:, self._joint_ids]
        dq_tustin, ddq_tustin = self._estimate_tustin_joint_dynamics(q)
        dq_parallel_tustin = self._serial_to_parallel_state(dq_tustin)
        ddq_parallel_tustin = self._serial_to_parallel_state(ddq_tustin)
        self._last_dq_tustin = dq_tustin.clone()
        self._last_ddq_tustin = ddq_tustin.clone()
        self._last_dq_parallel_tustin = dq_parallel_tustin.clone()
        self._last_ddq_parallel_tustin = ddq_parallel_tustin.clone()
        q_des = self._q_des

        if self._pd_control_mode == "implicit":
            q_target = self._compute_serial_target_from_bispace_action(q_des)
            self._robot.set_joint_position_target(q_target, joint_ids=self._joint_ids)
            self._update_implicit_pd_torque_logs(q_target, q, dq_raw)
            return

        # Serial PD torque control (disabled)
        # tau = self._pd.from_position(q_des, q, dq_raw)

        tau = torch.zeros_like(q)
        tau_comp = torch.zeros_like(q)
        if self.cfg.use_biarticular_hip_knee and len(self._hip_knee_pairs) > 0:
            hip_ids = [pair[0] for pair in self._hip_knee_pairs]
            knee_ids = [pair[1] for pair in self._hip_knee_pairs]
            M11 = torch.zeros((self.num_envs, len(self._hip_knee_pairs)), device=self.device, dtype=q.dtype)
            M12 = torch.zeros_like(M11)
            M21 = torch.zeros_like(M11)
            M22 = torch.zeros_like(M11)

            for pair_idx, (hip_id, knee_id) in enumerate(self._hip_knee_pairs):
                leg_name = str(self._joint_names[hip_id])[:2]
                q2 = q[:, knee_id]
                try:
                    Delta_M = get_Delta_M(
                        self._robot,
                        q2,
                        leg_name=leg_name,
                        env_id=None,
                    )
                except Exception:
                    Delta_M = torch.zeros((self.num_envs, 2, 2), device=self.device, dtype=q.dtype)

                M11[:, pair_idx] = Delta_M[:, 0, 0]
                M12[:, pair_idx] = Delta_M[:, 0, 1]
                M21[:, pair_idx] = Delta_M[:, 1, 0]
                M22[:, pair_idx] = Delta_M[:, 1, 1]

            action_delta = q_des - self._default_joint_pos
            qm_des = self._default_joint_pos[:, hip_ids] + action_delta[:, hip_ids]
            qb_des = self._default_joint_pos[:, hip_ids] + self._default_joint_pos[:, knee_ids] + action_delta[:, knee_ids]

            qm = q[:, hip_ids]
            qb = q[:, hip_ids] + q[:, knee_ids]
            dqm = dq_raw[:, hip_ids]
            dqb = dq_raw[:, hip_ids] + dq_raw[:, knee_ids]

            e_bi = torch.stack((qm_des - qm, qb_des - qb), dim=-1)
            dq_bi = torch.stack((dqm, dqb), dim=-1)
            tau_bi = self._compute_pd_torque(e_bi, dq_bi)
            tau_m = tau_bi[..., 0]
            tau_b = tau_bi[..., 1]

            # tau_comp_hip = (
            #     -self.Jm * ddq_tustin[:, hip_ids]
            #     - self.Jm * ddq_tustin[:, knee_ids]
            #     - self.Bm * dq_tustin[:, hip_ids]
            #     - self.Bm * dq_tustin[:, knee_ids]
            #     - M11 * ddq_tustin[:, hip_ids]
            #     - M12 * ddq_tustin[:, knee_ids]
            # ) 
            # tau_comp_knee = (
            #     - self.Jm * ddq_tustin[:, hip_ids] 
            #     - self.Bm * dq_tustin[:, hip_ids]
            #     - M22 * ddq_tustin[:, knee_ids]
            #     - M21 * ddq_tustin[:, hip_ids]
            # )

            if self._tau_compensation_mode == "prop":
                tau_comp_hip = (
                    - self.Jm * ddq_raw[:, knee_ids]
                    - self.Bm * dq_raw[:, knee_ids]
                    - M11 * ddq_raw[:, hip_ids]
                    - M12 * ddq_raw[:, knee_ids]
                )
                tau_comp_knee = (
                    - self.Jm * ddq_raw[:, hip_ids]
                    - self.Bm * dq_raw[:, hip_ids]
                    - M22 * ddq_raw[:, knee_ids]
                    - M21 * ddq_raw[:, hip_ids]
                )
            elif self._tau_compensation_mode == "dyn":
                tau_comp_hip = (
                    - self.Jm * ddq_raw[:, knee_ids]
                    - self.Bm * dq_raw[:, knee_ids]
                )
                tau_comp_knee = (
                    - self.Jm * ddq_raw[:, hip_ids]
                    - self.Bm * dq_raw[:, hip_ids]
                )
            elif self._tau_compensation_mode in ("none", "kinonly", "kin_only"):
                tau_comp_hip = torch.zeros_like(tau_m)
                tau_comp_knee = torch.zeros_like(tau_b)
            else:
                raise ValueError(
                    f"Unsupported tau_compensation_mode={self._tau_compensation_mode!r}. "
                    "Expected one of: none, dyn, prop."
                )


            tau[:, hip_ids] = tau_m + tau_b
            tau[:, knee_ids] = tau_b
            tau_comp[:, hip_ids] = tau_comp_hip
            tau_comp[:, knee_ids] = tau_comp_knee

            # Keep direct PD control for joints outside hip/knee biarticular pairs (e.g., HAA).
            bi_ids = sorted(set(hip_ids + knee_ids))
            remaining_ids = [i for i in range(q.shape[1]) if i not in bi_ids]
            if remaining_ids:
                tau[:, remaining_ids] = self._compute_pd_torque(
                    q_des[:, remaining_ids] - q[:, remaining_ids], dq_raw[:, remaining_ids]
                )
        else:
            raise RuntimeError(
                "Biarticular conversion is enabled but no hip/knee pairs were resolved. "
                "Check hip_joint_token/knee_joint_token and joint names."
            )

        if self.cfg.torque_limit is not None:
            tau = torch.clamp(tau, -self.cfg.torque_limit, self.cfg.torque_limit)

        # Logging-only torque without tau_comp:
        # - HAA (or non-paired joints): keep joint torque as-is.
        # - HIP (paired): convert to biarticular hip torque = tau_hip - tau_knee.
        # - KNEE (paired): keep tau_knee.
        tau_no_comp_log = tau.clone()
        if self.cfg.use_biarticular_hip_knee and len(self._hip_knee_pairs) > 0:
            hip_ids = [pair[0] for pair in self._hip_knee_pairs]
            knee_ids = [pair[1] for pair in self._hip_knee_pairs]
            tau_no_comp_log[:, hip_ids] = tau[:, hip_ids] - tau[:, knee_ids]
            tau_no_comp_log[:, knee_ids] = tau[:, knee_ids]
        self._last_tau_no_comp = tau_no_comp_log
        self._last_tau_comp = tau_comp.clone()

        if self._torque_queue is not None:
            self._torque_queue[:, :-1, :] = self._torque_queue[:, 1:, :].clone()
            self._torque_queue[:, -1, :] = tau
            # Queue axis is [oldest ... newest], so index=(max_delay-delay) selects per-env delayed torque.
            queue_indices = (self._torque_delay_max_steps - self._env_delay_steps).view(-1, 1, 1)
            queue_indices = queue_indices.expand(-1, 1, tau.shape[1])
            delayed_tau = torch.gather(self._torque_queue, dim=1, index=queue_indices).squeeze(1)
        else:
            delayed_tau = tau

        if self._dynamic_conversion_enabled:
            tau_to_apply = delayed_tau + tau_comp
        else:
            tau_to_apply = delayed_tau
        # Keep torque limit on tau (without compensation) only.
        # Do not clamp tau_to_apply after adding tau_comp.

        self._last_tau_applied = tau_to_apply.clone()
        self._robot.set_joint_effort_target(tau_to_apply, joint_ids=self._joint_ids)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        self._sample_pd_gains(env_ids)
        self._last_tau_no_comp[env_ids] = 0.0
        self._last_tau_comp[env_ids] = 0.0
        self._last_tau_applied[env_ids] = 0.0
        self._last_dq_tustin[env_ids] = 0.0
        self._last_ddq_tustin[env_ids] = 0.0
        self._last_dq_parallel_tustin[env_ids] = 0.0
        self._last_ddq_parallel_tustin[env_ids] = 0.0
        current_q = self._robot.data.joint_pos[env_ids][:, self._joint_ids]
        self._dq_filter_prev_input[env_ids] = current_q
        self._dq_filter_prev_output[env_ids] = 0.0
        self._ddq_filter_prev_input[env_ids] = 0.0
        self._ddq_filter_prev_output[env_ids] = 0.0
        if self._torque_delay_enabled:
            if self._torque_delay_max_steps > self._torque_delay_min_steps:
                sampled_delay = torch.randint(
                    self._torque_delay_min_steps,
                    self._torque_delay_max_steps + 1,
                    (env_ids.shape[0],),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                sampled_delay = torch.full(
                    (env_ids.shape[0],),
                    self._torque_delay_min_steps,
                    device=self.device,
                    dtype=torch.long,
                )
            self._env_delay_steps[env_ids] = sampled_delay
        else:
            self._env_delay_steps[env_ids] = 0
        if self._torque_queue is not None:
            self._torque_queue[env_ids] = 0.0

    def get_play_log_signals(self) -> dict[str, torch.Tensor | list[str] | list[int]]:
        """Return per-step tensors used by play-time CSV logging."""
        if torch.is_tensor(self._joint_ids):
            joint_ids = self._joint_ids.tolist()
        else:
            joint_ids = list(self._joint_ids)

        return {
            "joint_names": list(self._joint_names),
            "joint_ids": joint_ids,
            "joint_pos": self._robot.data.joint_pos[:, joint_ids],
            "joint_vel": self._robot.data.joint_vel[:, joint_ids],
            "joint_acc": self._robot.data.joint_acc[:, joint_ids],
            "joint_vel_raw": self._robot.data.joint_vel[:, joint_ids],
            "joint_acc_raw": self._robot.data.joint_acc[:, joint_ids],
            "joint_vel_tustin": self._last_dq_tustin,
            "joint_acc_tustin": self._last_ddq_tustin,
            "joint_vel_parallel_tustin": self._last_dq_parallel_tustin,
            "joint_acc_parallel_tustin": self._last_ddq_parallel_tustin,
            "target_joint_pos": self._q_des,
            "tau_no_comp": self._last_tau_no_comp,
            "tau_comp": self._last_tau_comp,
            "tau_applied_cmd": self._last_tau_applied,
            "applied_torque": self._robot.data.applied_torque[:, joint_ids],
            "base_lin_vel_b": self._robot.data.root_lin_vel_b,
            "base_ang_vel_b": self._robot.data.root_ang_vel_b,
            "base_quat_w": self._robot.data.root_quat_w,
            "base_pos_w": self._robot.data.root_pos_w,
            "contact_names": list(self._play_contact_names),
            "contact_state": torch.as_tensor(
                self._get_play_contact_state(self._play_log_env_id), device=self.device, dtype=self._q_des.dtype
            ).unsqueeze(0),
        }

    def step(self, action: torch.Tensor):
        self._pre_physics_step(action)

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self._record_play_log_sample(self._sim_step_counter)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
