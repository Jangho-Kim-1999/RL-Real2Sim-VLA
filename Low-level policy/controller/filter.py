"""Stateful low-pass filters for controller-side signal conditioning."""

from __future__ import annotations

import math

import torch


def derivative_filter(
    x: torch.Tensor,
    prev_input: torch.Tensor,
    prev_output: torch.Tensor,
    *,
    dt: float,
    cutoff_hz: float | None = None,
    tau: float | None = None,
) -> torch.Tensor:
    """Apply the Tustin-discretized derivative filter ``s / (tau s + 1)``.

    The caller owns the filter state. This function computes the next output:

    ``y[k] = b0*x[k] + b1*x[k-1] - a1*y[k-1]``

    where the coefficients come from the bilinear (Tustin) transform of
    ``s / (tau s + 1)``.
    """
    if dt <= 0.0:
        raise ValueError(f"derivative_filter dt must be positive. Received: {dt}")
    if tau is None:
        if cutoff_hz is None or cutoff_hz <= 0.0:
            raise ValueError(
                "derivative_filter requires either a positive tau or a positive cutoff_hz."
            )
        tau = 1.0 / (2.0 * math.pi * float(cutoff_hz))
    elif tau <= 0.0:
        raise ValueError(f"derivative_filter tau must be positive. Received: {tau}")

    k = 2.0 / float(dt)
    denom = float(tau) * k + 1.0
    b0 = k / denom
    b1 = -k / denom
    a1 = (1.0 - float(tau) * k) / denom
    return b0 * x + b1 * prev_input - a1 * prev_output


class LPF:
    """Tustin-discretized low-pass filter with first/second-order options.

    The second-order filter is implemented as two cascaded first-order sections
    with the same cutoff frequency, which keeps the implementation simple and
    stable for batched tensor signals.
    """

    def __init__(
        self,
        cutoff_hz: float,
        dt: float,
        state_shape: torch.Size | tuple[int, ...],
        device: torch.device | str,
        dtype: torch.dtype,
        order: int = 1,
    ):
        self.cutoff_hz = float(cutoff_hz)
        self.dt = float(dt)
        self.order = int(order)
        if self.order not in (1, 2):
            raise ValueError(f"LPF order must be 1 or 2. Received: {self.order}")
        if self.dt <= 0.0:
            raise ValueError(f"LPF dt must be positive. Received: {self.dt}")
        if self.cutoff_hz <= 0.0:
            raise ValueError(f"LPF cutoff_hz must be positive. Received: {self.cutoff_hz}")

        self._state_shape = tuple(state_shape)
        self._device = torch.device(device)
        self._dtype = dtype

        wc = 2.0 * math.pi * self.cutoff_hz
        k = 2.0 / self.dt
        denom = k + wc
        self._b0 = float(wc / denom)
        self._b1 = float(wc / denom)
        self._a1 = float((wc - k) / denom)

        self._prev_inputs = [torch.zeros(self._state_shape, device=self._device, dtype=self._dtype) for _ in range(self.order)]
        self._prev_outputs = [torch.zeros(self._state_shape, device=self._device, dtype=self._dtype) for _ in range(self.order)]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset filter state fully or for a subset of environments."""
        if env_ids is None:
            for prev_input, prev_output in zip(self._prev_inputs, self._prev_outputs):
                prev_input.zero_()
                prev_output.zero_()
            return

        env_ids = env_ids.to(device=self._device, dtype=torch.long)
        for prev_input, prev_output in zip(self._prev_inputs, self._prev_outputs):
            prev_input[env_ids] = 0.0
            prev_output[env_ids] = 0.0

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Filter batched input tensor and return the filtered signal."""
        y = x
        for stage in range(self.order):
            stage_input = y
            y = (
                self._b0 * stage_input
                + self._b1 * self._prev_inputs[stage]
                - self._a1 * self._prev_outputs[stage]
            )
            self._prev_inputs[stage] = stage_input.clone()
            self._prev_outputs[stage] = y.clone()
        return y
