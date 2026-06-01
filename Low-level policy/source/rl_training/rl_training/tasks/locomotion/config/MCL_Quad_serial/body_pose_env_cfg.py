# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import rl_training.tasks.locomotion.mdp as mdp
from rl_training.tasks.locomotion.M_rewards import RewardsCfg
from .flat_env_cfg import MCLQuadserialFlatEnvCfg


@configclass
class MCLQuadserialBodyPoseEnvCfg(MCLQuadserialFlatEnvCfg):
    """Flat standing task with in-place sinusoidal body velocity references and planted feet."""
    # joint_rom_violation termination thresholds (serial coordinates)
    # - q1: HIP angle, terminate if q1 > joint_rom_q1_upper_limit
    # - q2: KNEE angle, terminate if q2 < joint_rom_q2_lower_limit
    joint_rom_q1_upper_limit: float = 3.141592653589793
    joint_rom_q2_lower_limit: float = -3.141592653589793

    def __post_init__(self):
        super().__post_init__()
        reward_defaults = RewardsCfg()

        # flat_env_cfg prunes zero-weight placeholders; revive the terms that this task reconfigures.
        for term_name in (
            "track_stance_x_zero_exp",
            "track_stance_z_pos_ref_from_command_exp",
            "track_base_pitch_pos_ref_from_command_exp",
            "track_lin_vel_x_ref_exp",
            "track_lin_vel_z_ref_exp",
            "track_ang_vel_pitch_ref_exp",
            "lin_vel_x_l2",
            "lin_vel_y_l2",
            "ang_x_l2",
            "ang_vel_x_l2",
            "ang_vel_y_l2",
            "ang_vel_z_l2",
            "feet_contact_ratio",
            "feet_contact_without_cmd",
            "joint_torques_l2",
            "joint_power",
            "joint_rom_range_penalty",
            "joint_sync",
        ):
            if getattr(self.rewards, term_name, None) is None and hasattr(reward_defaults, term_name):
                setattr(self.rewards, term_name, getattr(reward_defaults, term_name))

        self.rewards.track_stance_x_zero_exp = reward_defaults.track_stance_x_zero_exp
        self.rewards.track_stance_z_pos_ref_from_command_exp = reward_defaults.track_stance_z_pos_ref_from_command_exp
        self.rewards.track_base_pitch_pos_ref_from_command_exp = reward_defaults.track_base_pitch_pos_ref_from_command_exp
        self.rewards.track_lin_vel_x_ref_exp = reward_defaults.track_lin_vel_x_ref_exp
        self.rewards.track_lin_vel_z_ref_exp = reward_defaults.track_lin_vel_z_ref_exp
        self.rewards.track_ang_vel_pitch_ref_exp = reward_defaults.track_ang_vel_pitch_ref_exp
        self.rewards.lin_vel_x_l2 = reward_defaults.lin_vel_x_l2
        self.rewards.lin_vel_y_l2 = reward_defaults.lin_vel_y_l2
        self.rewards.ang_x_l2 = reward_defaults.ang_x_l2
        self.rewards.ang_vel_x_l2 = reward_defaults.ang_vel_x_l2
        self.rewards.ang_vel_y_l2 = reward_defaults.ang_vel_y_l2
        self.rewards.ang_vel_z_l2 = reward_defaults.ang_vel_z_l2
        self.rewards.feet_contact_ratio = reward_defaults.feet_contact_ratio
        self.rewards.feet_contact_without_cmd = reward_defaults.feet_contact_without_cmd
        self.rewards.joint_rom_range_penalty = reward_defaults.joint_rom_range_penalty
        self.rewards.joint_sync = reward_defaults.joint_sync

        # ------------------------------Commands------------------------------
        self.commands.base_velocity = None
        self.commands.body_pose = mdp.SinusoidalBodyXZPitchCommandCfg(
            asset_name="robot",
            foot_body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
            resampling_time_range=(4.0, 6.0),
            z_pos_bias=0.3536,
            pitch_pos_bias=0.0,
            ranges=mdp.SinusoidalBodyXZPitchCommandCfg.Ranges(
                z_magnitude=(0.02, 0.08),
                z_frequency_hz=(0.2, 1.0),
                pitch_magnitude=(0.1, 0.4),
                pitch_frequency_hz=(0.2, 0.8),
            ),
            z_mode_probability=0.0,
        )

        # ------------------------------Observations------------------------------
        self.observations.policy.velocity_commands.params["command_name"] = "body_pose"
        self.observations.critic.velocity_commands.params["command_name"] = "body_pose"

        # ------------------------------Rewards------------------------------
        self.rewards.track_stance_x_zero_exp.weight = 7.0
        self.rewards.track_stance_x_zero_exp.params["std"] = 0.08

        self.rewards.track_stance_z_pos_ref_from_command_exp.weight = 5.0
        self.rewards.track_stance_z_pos_ref_from_command_exp.params["command_name"] = "body_pose"
        self.rewards.track_stance_z_pos_ref_from_command_exp.params["std"] = 0.08
        self.rewards.track_stance_z_pos_ref_from_command_exp.params["command_index"] = 1

        self.rewards.track_base_pitch_pos_ref_from_command_exp.weight = 10.0
        self.rewards.track_base_pitch_pos_ref_from_command_exp.params["command_name"] = "body_pose"
        self.rewards.track_base_pitch_pos_ref_from_command_exp.params["std"] = 0.08
        self.rewards.track_base_pitch_pos_ref_from_command_exp.params["command_index"] = 2

        self.rewards.track_lin_vel_x_ref_exp.weight = 1.0
        self.rewards.track_lin_vel_x_ref_exp.params["command_name"] = "body_pose"
        self.rewards.track_lin_vel_x_ref_exp.params["std"] = 0.08
        self.rewards.track_lin_vel_x_ref_exp.params["command_index"] = 0
        self.rewards.track_lin_vel_x_ref_exp.params["vel_axis"] = 0

        self.rewards.track_lin_vel_z_ref_exp.weight = 1.0
        self.rewards.track_lin_vel_z_ref_exp.params["command_name"] = "body_pose"
        self.rewards.track_lin_vel_z_ref_exp.params["std"] = 0.08
        self.rewards.track_lin_vel_z_ref_exp.params["command_index"] = 1
        self.rewards.track_lin_vel_z_ref_exp.params["vel_axis"] = 2

        self.rewards.track_ang_vel_pitch_ref_exp.weight = 10.0
        self.rewards.track_ang_vel_pitch_ref_exp.params["command_name"] = "body_pose"
        self.rewards.track_ang_vel_pitch_ref_exp.params["std"] = 0.08
        self.rewards.track_ang_vel_pitch_ref_exp.params["command_index"] = 2
        self.rewards.track_ang_vel_pitch_ref_exp.params["ang_axis"] = 1

        self.rewards.base_height_l2.weight = 0.0
        self.rewards.base_height_l2.params["target_height"] = 0.35
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.ang_x_l2.weight = -1.0
        self.rewards.ang_vel_x_l2.weight = -9.0
        self.rewards.ang_vel_y_l2.weight = 0.0
        self.rewards.ang_vel_z_l2.weight = -10.0

        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.lin_vel_x_l2.weight = 0.0
        self.rewards.lin_vel_y_l2.weight = -3.5

        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0

        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.inter_diagonal_load_balance.weight = 0.0

        self.rewards.feet_contact_ratio.weight = 0.0
        self.rewards.feet_contact_ratio.params["sensor_cfg"].body_names = [self.foot_link_name]
        if hasattr(reward_defaults, "feet_all_contact"):
            self.rewards.feet_all_contact = reward_defaults.feet_all_contact
            self.rewards.feet_all_contact.weight = 6.0
            self.rewards.feet_all_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_count_penalty.weight = 0.0
        self.rewards.feet_stuck_time_penalty.weight = 0.0
        self.rewards.feet_slide.weight = -6.0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.contact_forces.weight = 0 #-1.0e-3
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]

        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.joint_torques_no_comp_l2.weight = -2.0e-6
        self.rewards.joint_torques_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = -1.0e-6
        self.rewards.joint_power_no_comp.weight = -2.0e-5
        self.rewards.applied_torque_limits_no_comp.weight = -1.0e-4
        self.rewards.applied_torque_limits_no_comp.params["torque_limit"] = 60.0
        self.rewards.joint_power.weight = 0.0
        self.rewards.joint_pos_limits.weight = -8.0
        self.rewards.joint_rom_range_penalty.weight = -5.0
        self.rewards.joint_rom_range_penalty.params["q1_lower_limit"] = 0.0
        self.rewards.joint_rom_range_penalty.params["q1_upper_limit"] = 1.5707963267948966
        self.rewards.joint_rom_range_penalty.params["q2_lower_limit"] = 0.0
        self.rewards.joint_rom_range_penalty.params["q2_upper_limit"] = 3.141592653589793
        self.rewards.joint_deviation_l1.weight = -4.5
        self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = [".*HAA.*"]

        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.joint_mirror.weight = -5.0
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FL(HAA|HIP|KNEE).*", "FR(HAA|HIP|KNEE).*"],
            ["RL(HAA|HIP|KNEE).*", "RR(HAA|HIP|KNEE).*"],
        ]
        self.rewards.joint_sync.weight = 0.0
        self.rewards.joint_sync.params["joint_groups"] = [
            ["FL(HAA|HIP|KNEE).*", "FR(HAA|HIP|KNEE).*", "RL(HAA|HIP|KNEE).*", "RR(HAA|HIP|KNEE).*"],
        ]
        self.rewards.feet_contact_without_cmd.weight = 0.0
        # self.terminations.joint_rom_violation = None

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_link_name]), "threshold": 10.0},
        )

        if getattr(self.terminations, "joint_rom_violation", None) is not None:
            self.terminations.joint_rom_violation.params["q1_upper_limit"] = float(self.joint_rom_q1_upper_limit)
            self.terminations.joint_rom_violation.params["q2_lower_limit"] = float(self.joint_rom_q2_lower_limit)

        self.disable_zero_weight_rewards()
