# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import omni.log
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import RGBCommandCfg, UniformVelocityCommandStandingCfg


class UniformVelocityCommandStanding(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: UniformVelocityCommandStandingCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandStandingCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            lin_zero_ids = (self.vel_command_b[:, :2].norm(dim=1) <= 0.1).nonzero(as_tuple=False).flatten()
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            env_ids = env_ids[~torch.isin(env_ids, lin_zero_ids)]

            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0
        lin_zero_ids = self.vel_command_b[:, :2].norm(dim=1) <= 0.1
        self.vel_command_b[:, :2] *= (torch.norm(self.vel_command_b[:, :2], dim=1) > 0.1).unsqueeze(1)
        self.vel_command_b[:, 2] *= torch.abs(self.vel_command_b[:, 2]) > 0.1

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


class RGBCommand(CommandTerm):
    cfg: RGBCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: RGBCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # crete buffers to store the command
        # -- command: (R, G, B) values (0-1)
        self.rgb_command = torch.zeros(self.num_envs, 3, device=self.device)
        # -- rgb probability [0.5, 0.0, 0.5]
        self.rgb_prob = cfg.RGB_prob

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tRGB_prob: {self.cfg.RGB_prob}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired goal rgb. Shape is (num_envs, 3)."""
        return self.rgb_command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        num_resampled = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
        rand_rgb = torch.multinomial(
            torch.tensor(self.rgb_prob, device=self.device), num_resampled, replacement=True
        )
        self.rgb_command[env_ids] = torch.nn.functional.one_hot(rand_rgb, num_classes=3).float()
        self._resample_target_objects(env_ids, rand_rgb)

    def _resample_target_objects(self, env_ids: Sequence[int], rand_rgb: torch.Tensor):
        target_asset_names = self.cfg.target_asset_names
        target_pose_range = self.cfg.target_pose_range
        if target_asset_names is None or target_pose_range is None:
            return

        if isinstance(env_ids, slice):
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        elif torch.is_tensor(env_ids):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        range_list = [target_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_pos = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids_tensor), 3), device=self.device
        )

        yaw_range = target_pose_range.get("yaw", (0.0, 0.0))
        yaw_ranges = torch.tensor(yaw_range, device=self.device)
        rand_yaw = math_utils.sample_uniform(
            yaw_ranges[0], yaw_ranges[1], (len(env_ids_tensor),), device=self.device
        )
        yaw_delta = math_utils.quat_from_euler_xyz(
            torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw
        )

        env_origins = self._env.scene.env_origins[env_ids_tensor]
        inactive_pos = torch.tensor(self.cfg.inactive_target_pos, device=self.device).view(1, 1, 3)
        target_position_offsets = self.cfg.target_position_offsets

        for command_idx, asset_name in enumerate(target_asset_names):
            if asset_name is None:
                continue

            asset = self._env.scene[asset_name]
            root_states = asset.data.default_object_state[env_ids_tensor].clone()
            root_states[..., :3] = env_origins.unsqueeze(1) + inactive_pos

            active_mask = rand_rgb == command_idx
            if torch.any(active_mask):
                active_ids = active_mask.nonzero(as_tuple=False).flatten()
                if target_position_offsets is not None and target_position_offsets[command_idx] is not None:
                    target_offset = torch.tensor(
                        target_position_offsets[command_idx], device=self.device
                    ).view(1, 1, 3)
                else:
                    target_offset = torch.zeros((1, 1, 3), device=self.device)
                root_states[active_ids, ..., :3] = (
                    env_origins[active_ids].unsqueeze(1) + rand_pos[active_ids].unsqueeze(1)
                    + target_offset
                )
                root_states[active_ids, ..., 3:7] = math_utils.quat_mul(
                    root_states[active_ids, ..., 3:7],
                    yaw_delta[active_ids].unsqueeze(1),
                )

            asset.write_object_state_to_sim(root_states, env_ids=env_ids_tensor)
            self._update_target_lights(env_ids_tensor, asset_name, command_idx, active_mask, root_states)

    def _sample_target_light_intensity(self, command_idx: int) -> float:
        ranges = self.cfg.target_light_intensity_ranges
        if ranges is None or command_idx >= len(ranges) or ranges[command_idx] is None:
            intensity_range = (1000.0, 1000.0)
        else:
            intensity_range = ranges[command_idx]
        light_min = max(0.0, min(float(intensity_range[0]), float(intensity_range[1])))
        light_max = max(0.0, max(float(intensity_range[0]), float(intensity_range[1])))
        if light_min == light_max:
            return light_min
        return float(torch.empty((), device=self.device).uniform_(light_min, light_max).item())

    def _sample_target_light_color(self, command_idx: int):
        from pxr import Gf

        colors = self.cfg.target_light_colors
        if colors is None or command_idx >= len(colors) or colors[command_idx] is None:
            base_color = (1.0, 0.92, 0.82)
        else:
            base_color = colors[command_idx]
        jitter = max(0.0, float(self.cfg.target_light_color_jitter))
        if jitter > 0.0:
            noise = torch.empty(3, device=self.device).uniform_(-jitter, jitter).cpu().tolist()
        else:
            noise = (0.0, 0.0, 0.0)
        color = [min(1.0, max(0.0, float(channel) + float(delta))) for channel, delta in zip(base_color, noise)]
        return Gf.Vec3f(*color)

    def _update_target_lights(
        self,
        env_ids_tensor: torch.Tensor,
        asset_name: str,
        command_idx: int,
        active_mask: torch.Tensor,
        root_states: torch.Tensor,
    ) -> None:
        if not self.cfg.target_light_enabled:
            return

        import isaacsim.core.utils.prims as prim_utils

        for local_idx, env_id in enumerate(env_ids_tensor):
            light_path = f"/World/envs/env_{int(env_id.item())}/TargetFillLights/{asset_name}_fill_light"
            if prim_utils.is_prim_path_valid(light_path):
                prim_utils.delete_prim(light_path)
            if not bool(active_mask[local_idx]):
                continue

            position = root_states[local_idx, 0, :3].detach().cpu().tolist()
            position[2] += max(0.0, float(self.cfg.target_light_height))
            prim_utils.create_prim(
                prim_path=light_path,
                prim_type="SphereLight",
                translation=tuple(float(value) for value in position),
                attributes={
                    "inputs:intensity": self._sample_target_light_intensity(command_idx),
                    "inputs:radius": max(0.01, float(self.cfg.target_light_radius)),
                    "inputs:color": self._sample_target_light_color(command_idx),
                },
            )

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
