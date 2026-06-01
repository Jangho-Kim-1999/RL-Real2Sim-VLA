# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import rl_training.tasks.locomotion.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """The configuration of the command generator."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer.unsqueeze(-1).to(dtype=torch.float32)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """


class UniformBodyXPitchCommand(CommandTerm):
    """Command generator for standing body pose references [x_ref, pitch_ref]."""

    cfg: UniformBodyXPitchCommandCfg

    def __init__(self, cfg: UniformBodyXPitchCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        foot_ids, _ = self.robot.find_bodies(cfg.foot_body_names, preserve_order=True)
        self.foot_ids = torch.as_tensor(foot_ids, device=self.device, dtype=torch.long)
        self.command_buffer = torch.zeros(self.num_envs, 2, device=self.device)
        self.metrics["error_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "UniformBodyXPitchCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.command_buffer

    def _update_metrics(self):
        foot_center_w = self.robot.data.body_link_pos_w[:, self.foot_ids, :].mean(dim=1)
        delta_w = self.robot.data.root_pos_w - foot_center_w
        delta_b = math_utils.quat_apply_inverse(math_utils.yaw_quat(self.robot.data.root_quat_w), delta_w)
        _, pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        self.metrics["error_x"] = torch.abs(delta_b[:, 0] - self.command_buffer[:, 0])
        self.metrics["error_pitch"] = torch.abs(pitch - self.command_buffer[:, 1])

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.command_buffer[env_ids, 0] = r.uniform_(*self.cfg.ranges.x_ref)
        self.command_buffer[env_ids, 1] = r.uniform_(*self.cfg.ranges.pitch_ref)

    def _update_command(self):
        pass


@configclass
class UniformBodyXPitchCommandCfg(CommandTermCfg):
    """Configuration for standing body pose references [x_ref, pitch_ref]."""

    @configclass
    class Ranges:
        x_ref: tuple[float, float] = (-0.05, 0.05)
        pitch_ref: tuple[float, float] = (-0.12, 0.12)

    class_type: type = UniformBodyXPitchCommand

    asset_name: str = "robot"
    foot_body_names: tuple[str, str, str, str] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    ranges: Ranges = Ranges()


class UniformBodyXZPitchCommand(CommandTerm):
    """Command generator for standing body pose references [x_ref, z_ref, pitch_ref]."""

    cfg: "UniformBodyXZPitchCommandCfg"

    def __init__(self, cfg: "UniformBodyXZPitchCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        foot_ids, _ = self.robot.find_bodies(cfg.foot_body_names, preserve_order=True)
        self.foot_ids = torch.as_tensor(foot_ids, device=self.device, dtype=torch.long)
        self.command_buffer = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["error_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "UniformBodyXZPitchCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.command_buffer

    def _update_metrics(self):
        foot_center_w = self.robot.data.body_link_pos_w[:, self.foot_ids, :].mean(dim=1)
        delta_w = self.robot.data.root_pos_w - foot_center_w
        delta_b = math_utils.quat_apply_inverse(math_utils.yaw_quat(self.robot.data.root_quat_w), delta_w)
        _, pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        self.metrics["error_x"] = torch.abs(delta_b[:, 0] - self.command_buffer[:, 0])
        self.metrics["error_z"] = torch.abs(delta_b[:, 2] - self.command_buffer[:, 1])
        self.metrics["error_pitch"] = torch.abs(pitch - self.command_buffer[:, 2])

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.command_buffer[env_ids, 0] = r.uniform_(*self.cfg.ranges.x_ref)
        self.command_buffer[env_ids, 1] = r.uniform_(*self.cfg.ranges.z_ref)
        self.command_buffer[env_ids, 2] = r.uniform_(*self.cfg.ranges.pitch_ref)

    def _update_command(self):
        pass


@configclass
class UniformBodyXZPitchCommandCfg(CommandTermCfg):
    """Configuration for standing body pose references [x_ref, z_ref, pitch_ref]."""

    @configclass
    class Ranges:
        x_ref: tuple[float, float] = (-0.05, 0.05)
        z_ref: tuple[float, float] = (-0.06, 0.06)
        pitch_ref: tuple[float, float] = (-0.12, 0.12)

    class_type: type = UniformBodyXZPitchCommand

    asset_name: str = "robot"
    foot_body_names: tuple[str, str, str, str] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    ranges: Ranges = Ranges()


class UniformBodyRollPitchCommand(CommandTerm):
    """Command generator for body attitude references [roll_ref, pitch_ref]."""

    cfg: "UniformBodyRollPitchCommandCfg"

    def __init__(self, cfg: "UniformBodyRollPitchCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.command_buffer = torch.zeros(self.num_envs, 2, device=self.device)
        self.metrics["error_roll"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "UniformBodyRollPitchCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            f"\tRoll range: {self.cfg.ranges.roll_ref}\n"
            f"\tPitch range: {self.cfg.ranges.pitch_ref}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.command_buffer

    def _update_metrics(self):
        roll, pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        self.metrics["error_roll"] = torch.abs(roll - self.command_buffer[:, 0])
        self.metrics["error_pitch"] = torch.abs(pitch - self.command_buffer[:, 1])

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.command_buffer[env_ids, 0] = r.uniform_(*self.cfg.ranges.roll_ref)
        self.command_buffer[env_ids, 1] = r.uniform_(*self.cfg.ranges.pitch_ref)

    def _update_command(self):
        pass


@configclass
class UniformBodyRollPitchCommandCfg(CommandTermCfg):
    """Configuration for body attitude references [roll_ref, pitch_ref]."""

    @configclass
    class Ranges:
        roll_ref: tuple[float, float] = (-0.5, 0.5)
        pitch_ref: tuple[float, float] = (-0.85, 0.85)

    class_type: type = UniformBodyRollPitchCommand

    asset_name: str = "robot"
    ranges: Ranges = Ranges()


class UniformVxRollPitchCommand(CommandTerm):
    """Command generator for attitude references [vx_ref, roll_ref, pitch_ref] with vx fixed to 0."""

    cfg: "UniformVxRollPitchCommandCfg"

    def __init__(self, cfg: "UniformVxRollPitchCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.command_buffer = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["error_roll"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "UniformVxRollPitchCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            "\tvx_ref is fixed to 0.0\n"
            f"\tRoll range: {self.cfg.ranges.roll_ref}\n"
            f"\tPitch range: {self.cfg.ranges.pitch_ref}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.command_buffer

    def _update_metrics(self):
        roll, pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        self.metrics["error_roll"] = torch.abs(roll - self.command_buffer[:, 1])
        self.metrics["error_pitch"] = torch.abs(pitch - self.command_buffer[:, 2])

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.command_buffer[env_ids, 0] = 0.0
        self.command_buffer[env_ids, 1] = r.uniform_(*self.cfg.ranges.roll_ref)
        self.command_buffer[env_ids, 2] = r.uniform_(*self.cfg.ranges.pitch_ref)

    def _update_command(self):
        pass


@configclass
class UniformVxRollPitchCommandCfg(CommandTermCfg):
    """Configuration for attitude references [vx_ref, roll_ref, pitch_ref] with vx fixed to 0."""

    @configclass
    class Ranges:
        roll_ref: tuple[float, float] = (-0.5, 0.5)
        pitch_ref: tuple[float, float] = (-0.85, 0.85)

    class_type: type = UniformVxRollPitchCommand

    asset_name: str = "robot"
    ranges: Ranges = Ranges()


class SinusoidalBodyPitchCommand(CommandTerm):
    """Command generator for in-place sinusoidal body pitch references [x_ref, pitch_ref]."""

    cfg: "SinusoidalBodyPitchCommandCfg"

    def __init__(self, cfg: "SinusoidalBodyPitchCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        foot_ids, _ = self.robot.find_bodies(cfg.foot_body_names, preserve_order=True)
        self.foot_ids = torch.as_tensor(foot_ids, device=self.device, dtype=torch.long)
        self.command_buffer = torch.zeros(self.num_envs, 2, device=self.device)
        self.pitch_magnitude = torch.zeros(self.num_envs, device=self.device)
        self.pitch_frequency_hz = torch.zeros(self.num_envs, device=self.device)
        self.elapsed_time = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "SinusoidalBodyPitchCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            f"\tPitch magnitude range: {self.cfg.ranges.pitch_magnitude}\n"
            f"\tPitch frequency range [Hz]: {self.cfg.ranges.pitch_frequency_hz}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.command_buffer

    def _update_metrics(self):
        foot_center_w = self.robot.data.body_link_pos_w[:, self.foot_ids, :].mean(dim=1)
        delta_w = self.robot.data.root_pos_w - foot_center_w
        delta_b = math_utils.quat_apply_inverse(math_utils.yaw_quat(self.robot.data.root_quat_w), delta_w)
        _, pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        self.metrics["error_x"] = torch.abs(delta_b[:, 0] - self.command_buffer[:, 0])
        self.metrics["error_pitch"] = torch.abs(pitch - self.command_buffer[:, 1])

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.command_buffer[env_ids, 0] = r.uniform_(*self.cfg.ranges.x_ref)
        self.pitch_magnitude[env_ids] = r.uniform_(*self.cfg.ranges.pitch_magnitude)
        self.pitch_frequency_hz[env_ids] = r.uniform_(*self.cfg.ranges.pitch_frequency_hz)
        self.elapsed_time[env_ids] = 0.0
        self.command_buffer[env_ids, 1] = 0.0

    def _update_command(self):
        phase = 2.0 * torch.pi * self.pitch_frequency_hz * self.elapsed_time
        self.command_buffer[:, 1] = self.pitch_magnitude * torch.sin(phase)
        self.elapsed_time += float(self._env.step_dt)


@configclass
class SinusoidalBodyPitchCommandCfg(CommandTermCfg):
    """Configuration for in-place sinusoidal body pitch references [x_ref, pitch_ref]."""

    @configclass
    class Ranges:
        x_ref: tuple[float, float] = (0.0, 0.0)
        pitch_magnitude: tuple[float, float] = (0.04, 0.12)
        pitch_frequency_hz: tuple[float, float] = (0.2, 0.6)

    class_type: type = SinusoidalBodyPitchCommand

    asset_name: str = "robot"
    foot_body_names: tuple[str, str, str, str] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    ranges: Ranges = Ranges()


class SinusoidalBodyZPitchCommand(CommandTerm):
    """Command generator for in-place sinusoidal body references [z_ref, pitch_ref]."""

    cfg: "SinusoidalBodyZPitchCommandCfg"

    def __init__(self, cfg: "SinusoidalBodyZPitchCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        foot_ids, _ = self.robot.find_bodies(cfg.foot_body_names, preserve_order=True)
        self.foot_ids = torch.as_tensor(foot_ids, device=self.device, dtype=torch.long)
        self.command_buffer = torch.zeros(self.num_envs, 2, device=self.device)
        self.z_magnitude = torch.zeros(self.num_envs, device=self.device)
        self.z_frequency_hz = torch.zeros(self.num_envs, device=self.device)
        self.pitch_magnitude = torch.zeros(self.num_envs, device=self.device)
        self.pitch_frequency_hz = torch.zeros(self.num_envs, device=self.device)
        self.elapsed_time = torch.zeros(self.num_envs, device=self.device)
        # 0: z-only, 1: pitch-only
        self.active_mode = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.metrics["error_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "SinusoidalBodyZPitchCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            f"\tZ magnitude range: {self.cfg.ranges.z_magnitude}\n"
            f"\tZ frequency range [Hz]: {self.cfg.ranges.z_frequency_hz}\n"
            f"\tPitch magnitude range: {self.cfg.ranges.pitch_magnitude}\n"
            f"\tPitch frequency range [Hz]: {self.cfg.ranges.pitch_frequency_hz}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.command_buffer

    def _update_metrics(self):
        foot_center_w = self.robot.data.body_link_pos_w[:, self.foot_ids, :].mean(dim=1)
        delta_w = self.robot.data.root_pos_w - foot_center_w
        delta_b = math_utils.quat_apply_inverse(math_utils.yaw_quat(self.robot.data.root_quat_w), delta_w)
        _, pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        self.metrics["error_z"] = torch.abs(delta_b[:, 2] - self.command_buffer[:, 0])
        self.metrics["error_pitch"] = torch.abs(pitch - self.command_buffer[:, 1])

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.z_magnitude[env_ids] = r.uniform_(*self.cfg.ranges.z_magnitude)
        self.z_frequency_hz[env_ids] = r.uniform_(*self.cfg.ranges.z_frequency_hz)
        self.pitch_magnitude[env_ids] = r.uniform_(*self.cfg.ranges.pitch_magnitude)
        self.pitch_frequency_hz[env_ids] = r.uniform_(*self.cfg.ranges.pitch_frequency_hz)
        z_prob = float(self.cfg.z_mode_probability)
        self.active_mode[env_ids] = (torch.rand(len(env_ids), device=self.device) >= z_prob).long()
        self.elapsed_time[env_ids] = 0.0
        self.command_buffer[env_ids, 0] = 0.0
        self.command_buffer[env_ids, 1] = 0.0

    def _update_command(self):
        z_phase = 2.0 * torch.pi * self.z_frequency_hz * self.elapsed_time
        pitch_phase = 2.0 * torch.pi * self.pitch_frequency_hz * self.elapsed_time
        z_wave = self.z_magnitude * torch.sin(z_phase)
        pitch_wave = self.pitch_magnitude * torch.sin(pitch_phase)
        z_mask = (self.active_mode == 0).float()
        pitch_mask = (self.active_mode == 1).float()
        self.command_buffer[:, 0] = z_wave * z_mask
        self.command_buffer[:, 1] = pitch_wave * pitch_mask
        self.elapsed_time += float(self._env.step_dt)


@configclass
class SinusoidalBodyZPitchCommandCfg(CommandTermCfg):
    """Configuration for in-place sinusoidal body references [z_ref, pitch_ref]."""

    @configclass
    class Ranges:
        z_magnitude: tuple[float, float] = (0.02, 0.06)
        z_frequency_hz: tuple[float, float] = (0.2, 0.8)
        pitch_magnitude: tuple[float, float] = (0.04, 0.12)
        pitch_frequency_hz: tuple[float, float] = (0.2, 0.6)

    class_type: type = SinusoidalBodyZPitchCommand

    asset_name: str = "robot"
    foot_body_names: tuple[str, str, str, str] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    ranges: Ranges = Ranges()
    z_mode_probability: float = 0.5


class SinusoidalBodyXZPitchCommand(CommandTerm):
    """Command generator for sinusoidal body-velocity references [vx_ref, vz_ref, pitch_rate_ref].

    Note:
        The configured amplitudes/frequencies are interpreted as position-domain sinusoid parameters
        and converted to velocity-domain commands via time derivative.
    """

    cfg: "SinusoidalBodyXZPitchCommandCfg"

    def __init__(self, cfg: "SinusoidalBodyXZPitchCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        foot_ids, _ = self.robot.find_bodies(cfg.foot_body_names, preserve_order=True)
        self.foot_ids = torch.as_tensor(foot_ids, device=self.device, dtype=torch.long)
        self.command_buffer = torch.zeros(self.num_envs, 3, device=self.device)
        # Position-domain references paired with the velocity command:
        # [x_ref, z_ref, pitch_ref] where x_ref is fixed to 0.0.
        self.pos_ref_buffer = torch.zeros(self.num_envs, 3, device=self.device)
        self.z_magnitude = torch.zeros(self.num_envs, device=self.device)
        self.z_frequency_hz = torch.zeros(self.num_envs, device=self.device)
        self.pitch_magnitude = torch.zeros(self.num_envs, device=self.device)
        self.pitch_frequency_hz = torch.zeros(self.num_envs, device=self.device)
        self.elapsed_time = torch.zeros(self.num_envs, device=self.device)
        # 0: z-only, 1: pitch-only
        self.active_mode = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.metrics["error_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "SinusoidalBodyXZPitchCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            "\tX command is fixed to 0.0\n"
            f"\tZ position bias: {self.cfg.z_pos_bias}\n"
            f"\tPitch position bias: {self.cfg.pitch_pos_bias}\n"
            f"\tZ magnitude range: {self.cfg.ranges.z_magnitude}\n"
            f"\tZ frequency range [Hz]: {self.cfg.ranges.z_frequency_hz}\n"
            f"\tPitch magnitude range: {self.cfg.ranges.pitch_magnitude}\n"
            f"\tPitch frequency range [Hz]: {self.cfg.ranges.pitch_frequency_hz}\n"
            f"\tZ-only mode probability: {self.cfg.z_mode_probability}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.command_buffer

    def _update_metrics(self):
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        self.metrics["error_x"] = torch.abs(root_lin_vel_b[:, 0] - self.command_buffer[:, 0])
        self.metrics["error_z"] = torch.abs(root_lin_vel_b[:, 2] - self.command_buffer[:, 1])
        self.metrics["error_pitch"] = torch.abs(root_ang_vel_b[:, 1] - self.command_buffer[:, 2])

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.z_magnitude[env_ids] = r.uniform_(*self.cfg.ranges.z_magnitude)
        self.z_frequency_hz[env_ids] = r.uniform_(*self.cfg.ranges.z_frequency_hz)
        self.pitch_magnitude[env_ids] = r.uniform_(*self.cfg.ranges.pitch_magnitude)
        self.pitch_frequency_hz[env_ids] = r.uniform_(*self.cfg.ranges.pitch_frequency_hz)
        z_prob = float(self.cfg.z_mode_probability)
        self.active_mode[env_ids] = (torch.rand(len(env_ids), device=self.device) >= z_prob).long()
        self.elapsed_time[env_ids] = 0.0
        self.command_buffer[env_ids, :] = 0.0
        self.pos_ref_buffer[env_ids, 0] = 0.0
        self.pos_ref_buffer[env_ids, 1] = float(self.cfg.z_pos_bias)
        self.pos_ref_buffer[env_ids, 2] = float(self.cfg.pitch_pos_bias)

    def _update_command(self):
        z_omega = 2.0 * torch.pi * self.z_frequency_hz
        pitch_omega = 2.0 * torch.pi * self.pitch_frequency_hz
        z_phase = z_omega * self.elapsed_time
        pitch_phase = pitch_omega * self.elapsed_time
        # Position-domain sinusoidal references.
        z_pos_wave = self.z_magnitude * torch.sin(z_phase)
        pitch_pos_wave = self.pitch_magnitude * torch.sin(pitch_phase)
        # Velocity commands from position-domain sinusoids: d/dt(A*sin(wt)) = A*w*cos(wt).
        z_vel_wave = self.z_magnitude * z_omega * torch.cos(z_phase)
        pitch_rate_wave = self.pitch_magnitude * pitch_omega * torch.cos(pitch_phase)
        z_mask = (self.active_mode == 0).float()
        pitch_mask = (self.active_mode == 1).float()
        self.pos_ref_buffer[:, 0] = 0.0
        self.pos_ref_buffer[:, 1] = float(self.cfg.z_pos_bias) + z_pos_wave * z_mask
        self.pos_ref_buffer[:, 2] = float(self.cfg.pitch_pos_bias) + pitch_pos_wave * pitch_mask
        self.command_buffer[:, 0] = 0.0
        self.command_buffer[:, 1] = z_vel_wave * z_mask
        self.command_buffer[:, 2] = pitch_rate_wave * pitch_mask
        self.elapsed_time += float(self._env.step_dt)


@configclass
class SinusoidalBodyXZPitchCommandCfg(CommandTermCfg):
    """Configuration for in-place sinusoidal body-velocity references [vx_ref, vz_ref, pitch_rate_ref]."""

    @configclass
    class Ranges:
        z_magnitude: tuple[float, float] = (0.02, 0.06)
        z_frequency_hz: tuple[float, float] = (0.2, 0.8)
        pitch_magnitude: tuple[float, float] = (0.05, 0.25)
        pitch_frequency_hz: tuple[float, float] = (0.2, 0.8)

    class_type: type = SinusoidalBodyXZPitchCommand

    asset_name: str = "robot"
    foot_body_names: tuple[str, str, str, str] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    ranges: Ranges = Ranges()
    z_pos_bias: float = 0.3536
    pitch_pos_bias: float = 0.0
    z_mode_probability: float = 0.5
