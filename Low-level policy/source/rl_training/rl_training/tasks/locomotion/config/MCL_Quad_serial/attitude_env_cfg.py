# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

import rl_training.tasks.locomotion.mdp as mdp
from rl_training.tasks.locomotion.M_rewards import RewardsCfg
from .flat_env_cfg import MCLQuadserialFlatEnvCfg


@configclass
class MCLQuadserialAttitudeEnvCfg(MCLQuadserialFlatEnvCfg):
    """Attitude-tracking task (roll/pitch) adapted from SPI-style reward/obs."""

    def __post_init__(self):
        super().__post_init__()
        reward_defaults = RewardsCfg()

        def _set_weight_if_exists(term_name: str, weight: float) -> None:
            term = getattr(self.rewards, term_name, None)
            if term is not None:
                term.weight = float(weight)

        # Revive reward placeholders that may have been pruned in parent cfg.
        for term_name in (
            "orientation_control_exp",
            "base_height_exp",
            "feet_distance_y_exp",
            "lin_vel_x_l2",
            "lin_vel_y_l2",
            "lin_vel_z_l2",
            "joint_torques_no_comp_l2",
            "joint_acc_l2",
            "action_rate_l2",
            "feet_slide",
            "joint_pos_limits",
            "joint_sync",
            "undesired_contacts",
        ):
            if getattr(self.rewards, term_name, None) is None and hasattr(reward_defaults, term_name):
                setattr(self.rewards, term_name, getattr(reward_defaults, term_name))

        # ------------------------------Commands------------------------------
        self.commands.base_velocity = None
        self.commands.attitude = mdp.UniformVxRollPitchCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 5.0),
            ranges=mdp.UniformVxRollPitchCommandCfg.Ranges(
                roll_ref=(-0.5, 0.5),
                pitch_ref=(-0.85, 0.85),
            ),
        )

        # ------------------------------Observations------------------------------
        # SPI actor: [base_ang_vel, projected_gravity, command_rp, dof_pos, dof_vel, actions]
        self.observations.policy.velocity_commands.params["command_name"] = "attitude"
        self.observations.critic.velocity_commands.params["command_name"] = "attitude"
        self.observations.policy.base_lin_vel = None  # type: ignore
        self.observations.policy.height_scan = None  # type: ignore
        self.observations.critic.height_scan = None  # type: ignore

        # ------------------------------Rewards------------------------------
        # Main tracking rewards
        self.rewards.orientation_control_exp = reward_defaults.orientation_control_exp
        self.rewards.orientation_control_exp.weight = 15.0
        self.rewards.orientation_control_exp.params["command_name"] = "attitude"
        self.rewards.orientation_control_exp.params["std"] = 0.25
        self.rewards.orientation_control_exp.params["roll_index"] = 1
        self.rewards.orientation_control_exp.params["pitch_index"] = 2

        self.rewards.base_height_exp = reward_defaults.base_height_exp
        self.rewards.base_height_exp.weight = 1.0
        self.rewards.base_height_exp.params["target_height"] = 0.34
        self.rewards.base_height_exp.params["std"] = 0.3

        # SPI-style penalty and auxiliary terms
        self.rewards.undesired_contacts.weight = -0.1
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.undesired_contacts.params["threshold"] = 0.1

        self.rewards.feet_distance_y_exp = reward_defaults.feet_distance_y_exp
        self.rewards.feet_distance_y_exp.weight = 0.5
        self.rewards.feet_distance_y_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.3
        self.rewards.feet_distance_y_exp.params["std"] = 0.25

        self.rewards.lin_vel_x_l2.weight = -5.0
        self.rewards.lin_vel_y_l2.weight = -0.8
        self.rewards.lin_vel_z_l2.weight = -0.7
        self.rewards.joint_torques_no_comp_l2.weight = -2.0e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.joint_pos_limits.params["soft_ratio"] = 0.9
        self.rewards.joint_pos_limits.params["margin_ratio"] = 0.0
        self.rewards.joint_pos_limits.params["power"] = 2.0
        self.rewards.joint_pos_limits.params["bispace_limit_overrides"] = {
            "QM": (0.0, 3.0),
            "QB": (1.0, 3.0),
        }
        self.rewards.joint_sync = reward_defaults.joint_sync
        self.rewards.joint_sync.weight = -1.0
        self.rewards.joint_sync.params["joint_groups"] = [
            ["FL(HAA|HIP|KNEE).*", "FR(HAA|HIP|KNEE).*"],
            ["RL(HAA|HIP|KNEE).*", "RR(HAA|HIP|KNEE).*"],
        ]

        # Disable non-attitude task terms
        _set_weight_if_exists("base_height_l2", 0.0)
        _set_weight_if_exists("flat_orientation_l2", 0.0)
        _set_weight_if_exists("ang_vel_xy_l2", 0.0)
        _set_weight_if_exists("track_lin_vel_xy_exp", 0.0)
        _set_weight_if_exists("track_ang_vel_z_exp", 0.0)
        _set_weight_if_exists("stand_still", 0.0)
        _set_weight_if_exists("feet_air_time", 0.0)
        _set_weight_if_exists("feet_air_time_variance", 0.0)
        _set_weight_if_exists("feet_gait", 0.0)
        _set_weight_if_exists("feet_height", 0.0)
        _set_weight_if_exists("feet_height_body", 0.0)
        _set_weight_if_exists("feet_contact_count_penalty", 0.0)
        _set_weight_if_exists("feet_stuck_time_penalty", 0.0)
        _set_weight_if_exists("feet_contact_without_cmd", 0.0)
        _set_weight_if_exists("stand_still_without_cmd", 0.0)
        _set_weight_if_exists("joint_pos_penalty", 0.0)
        _set_weight_if_exists("hipx_joint_pos_penalty", 0.0)
        _set_weight_if_exists("hipy_joint_pos_penalty", 0.0)
        _set_weight_if_exists("knee_joint_pos_penalty", 0.0)
        _set_weight_if_exists("joint_power", 0.0)
        _set_weight_if_exists("joint_power_no_comp", 0.0)
        _set_weight_if_exists("joint_torques_l2", 0.0)
        _set_weight_if_exists("applied_torque_limits", 0.0)
        _set_weight_if_exists("applied_torque_limits_no_comp", 0.0)
        _set_weight_if_exists("joint_deviation_l1", 0.0)
        _set_weight_if_exists("joint_mirror", 0.0)
        _set_weight_if_exists("inter_diagonal_load_balance", 0.0)

        self.disable_zero_weight_rewards()
