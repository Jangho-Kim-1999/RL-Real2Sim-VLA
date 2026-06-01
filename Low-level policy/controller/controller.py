"""PD controller for position-to-torque conversion."""

from __future__ import annotations

import torch


class PDController:
    """Computes torque from position error and velocity."""

    def __init__(self, kp: float = 50.0, kd: float = 1.5):
        self.kp = float(kp)
        self.kd = float(kd)

    def compute(self, position_error: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Return torque = kp * position_error - kd * velocity."""
        return self.kp * position_error - self.kd * velocity

    def from_position(self, q_des: torch.Tensor, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """Return torque from desired/current position and current velocity."""
        return self.compute(q_des - q, dq)


class ForceObserver:
    """Parallel-space force observer residual calculator for one leg.

    The observer currently computes the residual:

    ``tau_input - h_parallel - M_parallel @ ddq_parallel_tustin - Bm * dq_parallel_tustin``

    where:
    - ``tau_input[..., 3] = [tau0, taum, taub]`` and must exclude ``tau_comp``
    - ``q_input[..., 3] = [q0, qm, qb]``
    - ``h_parallel[..., 3]`` is the parallel-space ``C(q,dq)dq + G(q)`` term
    - ``M_parallel[..., 3, 3]`` is the actual 3-DOF parallel inertia matrix
    - ``dq_parallel_tustin[..., 3]`` and ``ddq_parallel_tustin[..., 3]`` are
      Tustin-derived states in parallel coordinates
    - ``motor_damping`` is a scalar or per-joint damping coefficient
    """

    def __init__(self):
        self._last_input: dict[str, torch.Tensor] | None = None

    @property
    def last_input(self) -> dict[str, torch.Tensor] | None:
        """Latest received observer input."""
        return self._last_input

    def reset(self) -> None:
        """Clear the last received input."""
        self._last_input = None

    def observe(
        self,
        tau_input: torch.Tensor,
        q_input: torch.Tensor,
        h_parallel: torch.Tensor,
        M_parallel: torch.Tensor,
        dq_parallel_tustin: torch.Tensor,
        ddq_parallel_tustin: torch.Tensor,
        motor_damping: torch.Tensor | float,
    ) -> dict[str, torch.Tensor]:
        """Receive observer inputs and compute the parallel-space residual.

        Args:
            tau_input: Tensor with trailing dimension 3 ordered as
                ``[tau0, taum, taub]``. This must exclude ``tau_comp``.
            q_input: Tensor with trailing dimension 3 ordered as
                ``[q0, qm, qb]``.
            h_parallel: Tensor with trailing dimension 3 containing the
                parallel-space dynamic vector ``C(q,dq)dq + G(q)``.
            M_parallel: Tensor with trailing dimensions ``(3, 3)``.
            dq_parallel_tustin: Tensor with trailing dimension 3 ordered as
                ``[dq0, dqm, dqb]``.
            ddq_parallel_tustin: Tensor with trailing dimension 3 ordered as
                ``[ddq0, ddqm, ddqb]``.
            motor_damping: Scalar or tensor broadcastable to the shape of
                ``dq_parallel_tustin``.
        """
        if tau_input.shape != q_input.shape:
            raise ValueError(
                f"ForceObserver inputs must have identical shapes; got "
                f"tau={tuple(tau_input.shape)}, q={tuple(q_input.shape)}."
            )
        if tau_input.shape[-1] != 3:
            raise ValueError(
                "ForceObserver expects per-leg inputs ordered as "
                "[tau0, taum, taub] and [q0, qm, qb] with trailing dimension 3; "
                f"got {tuple(tau_input.shape)}."
            )
        if h_parallel.shape != tau_input.shape:
            raise ValueError(
                "ForceObserver expects h_parallel to have the same shape as tau_input; "
                f"got h={tuple(h_parallel.shape)}, tau={tuple(tau_input.shape)}."
            )
        if dq_parallel_tustin.shape != tau_input.shape or ddq_parallel_tustin.shape != tau_input.shape:
            raise ValueError(
                "ForceObserver expects dq_parallel_tustin and ddq_parallel_tustin to match tau_input shape; "
                f"got dq={tuple(dq_parallel_tustin.shape)}, ddq={tuple(ddq_parallel_tustin.shape)}, tau={tuple(tau_input.shape)}."
            )
        if M_parallel.shape[:-2] != tau_input.shape[:-1] or M_parallel.shape[-2:] != (3, 3):
            raise ValueError(
                "ForceObserver expects M_parallel shape [..., 3, 3] matching tau_input batch dims; "
                f"got M={tuple(M_parallel.shape)}, tau={tuple(tau_input.shape)}."
            )

        if torch.is_tensor(motor_damping):
            damping_term = motor_damping * dq_parallel_tustin
        else:
            damping_term = float(motor_damping) * dq_parallel_tustin

        inertia_term = torch.matmul(M_parallel, ddq_parallel_tustin.unsqueeze(-1)).squeeze(-1)
        residual = tau_input - h_parallel - inertia_term - damping_term

        received_input = {
            "tau_input": tau_input,
            "q_input": q_input,
            "h_parallel": h_parallel,
            "M_parallel": M_parallel,
            "dq_parallel_tustin": dq_parallel_tustin,
            "ddq_parallel_tustin": ddq_parallel_tustin,
            "tau0": tau_input[..., 0],
            "taum": tau_input[..., 1],
            "taub": tau_input[..., 2],
            "q0": q_input[..., 0],
            "qm": q_input[..., 1],
            "qb": q_input[..., 2],
            "inertia_term": inertia_term,
            "damping_term": damping_term,
            "residual": residual,
        }
        self._last_input = received_input
        return received_input

    def __call__(
        self,
        tau_input: torch.Tensor,
        q_input: torch.Tensor,
        h_parallel: torch.Tensor,
        M_parallel: torch.Tensor,
        dq_parallel_tustin: torch.Tensor,
        ddq_parallel_tustin: torch.Tensor,
        motor_damping: torch.Tensor | float,
    ) -> dict[str, torch.Tensor]:
        """Callable alias of :meth:`observe`."""
        return self.observe(
            tau_input,
            q_input,
            h_parallel,
            M_parallel,
            dq_parallel_tustin,
            ddq_parallel_tustin,
            motor_damping,
        )



# python scripts/play.py \
#   --task Attitude-MCLQuad-serial \
#   --checkpoint logs/rsl_rl/MCLrobotics_MCLQuadserial_attitude/pitch_Kinonly_NoRand/model_1500.pt \
#   --num_envs 1 \
#   --attitude-pitch-amp 0.06 \
#   --attitude-pitch-freq 3.0 \
#   --traj-initial-stop 2.0 \
#   --attitude-command-duration-s 30.0 \
#   --play-duration-s 35.0 \
#   --hip-armature-scale 2.0 \
#   --hip-viscous-friction-scale 2.0 \
#   --tau-compensation-mode prop \
#   --play-rand-scope whole \
#   --log-play-signals-path logs/rsl_rl/MCLrobotics_MCLQuadserial_attitude/pitch2hz_0p08_check/Prop_Rand.csv
