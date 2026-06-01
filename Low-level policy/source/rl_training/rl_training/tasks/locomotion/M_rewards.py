# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.locomotion.mdp as mdp


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # General
    is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)

    # Root penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    lin_vel_x_l2 = RewTerm(func=mdp.lin_vel_x_l2, weight=0.0)
    lin_vel_y_l2 = RewTerm(func=mdp.lin_vel_y_l2, weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    ang_vel_x_l2 = RewTerm(func=mdp.ang_vel_x_l2, weight=0.0)
    ang_x_l2 = RewTerm(func=mdp.ang_x_l2, weight=0.0)
    ang_vel_y_l2 = RewTerm(func=mdp.ang_vel_y_l2, weight=0.0)
    ang_vel_z_l2 = RewTerm(func=mdp.ang_vel_z_l2, weight=0.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.0,
        },
    )
    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "soft_ratio": 1.0,
            "margin_ratio": 0.0,
            "power": 2.0,
            "limit_overrides": None,
            "bispace_limit_overrides": None,
        },
    )
    # Joint penalties
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_torques_no_comp_l2 = RewTerm(
        func=mdp.joint_torques_no_comp_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )

    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    def create_joint_deviation_l1_rewterm(self, attr_name, weight, joint_names_pattern):
        rew_term = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=weight,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_pattern)},
        )
        setattr(self, attr_name, rew_term)

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "soft_ratio": 1.0,
            "margin_ratio": 0.0,
            "power": 2.0,
            "limit_overrides": None,
            "bispace_limit_overrides": None,
        },
    )
    joint_rom_range_penalty = RewTerm(
        func=mdp.joint_rom_range_penalty,
        weight=0.0,
        params={
            "q1_lower_limit": 0.0,
            "q1_upper_limit": 1.5707963267948966,
            "q2_lower_limit": 0.0,
            "q2_upper_limit": 3.141592653589793,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0},
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    joint_power_no_comp = RewTerm(
        func=mdp.joint_power_no_comp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    stand_still_without_cmd = RewTerm(
        func=mdp.stand_still_without_cmd,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    hipx_joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    hipy_joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    knee_joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )

    action_mirror = RewTerm(
        func=mdp.action_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )

    action_sync = RewTerm(
        func=mdp.action_sync,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_groups": [
                ["FR_hip_joint", "FL_hip_joint", "RL_hip_joint", "RR_hip_joint"],
                ["FR_thigh_joint", "FL_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
                ["FR_calf_joint", "FL_calf_joint", "RL_calf_joint", "RR_calf_joint"],
            ],
        },
    )
    joint_sync = RewTerm(
        func=mdp.joint_sync,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_groups": [
                ["FL(HAA|HIP|KNEE).*", "FR(HAA|HIP|KNEE).*", "RL(HAA|HIP|KNEE).*", "RR(HAA|HIP|KNEE).*"],
            ],
        },
    )

    # Action penalties
    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    applied_torque_limits_no_comp = RewTerm(
        func=mdp.applied_torque_limits_no_comp,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "torque_limit": None},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    # smoothness_1 = RewTerm(func=mdp.smoothness_1, weight=0.0)  # Same as action_rate_l2
    # smoothness_2 = RewTerm(func=mdp.smoothness_2, weight=0.0)  # Unvaliable now

    # Contact sensor
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 1.0,
        },
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 300.0},
    )
    diagonal_grf_balance = RewTerm(
        func=mdp.diagonal_grf_balance,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("ground_reaction_forces"),
            "paired_body_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "pair_load_threshold": 10.0,
            "eps": 1.0,
        },
    )
    inter_diagonal_load_balance = RewTerm(
        func=mdp.inter_diagonal_load_balance,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("ground_reaction_forces"),
            "paired_body_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "total_load_threshold": 10.0,
            "eps": 1.0,
        },
    )

    # Velocity-tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )
    track_stance_x_zero_exp = RewTerm(
        func=mdp.track_stance_x_zero_exp,
        weight=0.0,
        params={
            "std": 0.03,
            "foot_body_names": ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    track_stance_z_pos_ref_from_command_exp = RewTerm(
        func=mdp.track_stance_z_pos_ref_from_command_exp,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "std": 0.03,
            "command_index": 1,
            "foot_body_names": ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    track_base_pitch_pos_ref_from_command_exp = RewTerm(
        func=mdp.track_base_pitch_pos_ref_from_command_exp,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "std": 0.08,
            "command_index": 2,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    orientation_control_exp = RewTerm(
        func=mdp.orientation_control_exp,
        weight=0.0,
        params={
            "command_name": "attitude",
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    base_height_exp = RewTerm(
        func=mdp.base_height_exp,
        weight=0.0,
        params={
            "std": 0.25,
            "target_height": 0.35,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    track_lin_vel_x_ref_exp = RewTerm(
        func=mdp.track_lin_vel_axis_ref_exp,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "std": 0.08,
            "command_index": 0,
            "vel_axis": 0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    track_lin_vel_z_ref_exp = RewTerm(
        func=mdp.track_lin_vel_axis_ref_exp,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "std": 0.08,
            "command_index": 1,
            "vel_axis": 2,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    track_ang_vel_pitch_ref_exp = RewTerm(
        func=mdp.track_ang_vel_axis_ref_exp,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "std": 0.15,
            "command_index": 2,
            "ang_axis": 1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    track_stance_x_ref_exp = RewTerm(
        func=mdp.track_stance_x_ref_exp,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "std": 0.03,
            "foot_body_names": ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    track_stance_z_ref_exp = RewTerm(
        func=mdp.track_stance_z_ref_exp,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "std": 0.03,
            "command_index": 0,
            "foot_body_names": ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    track_base_yaw_ref_exp = RewTerm(
        func=mdp.track_base_yaw_ref_exp,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "std": 0.08,
            "command_index": 2,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    track_base_pitch_ref_exp = RewTerm(
        func=mdp.track_base_pitch_ref_exp,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "std": 0.08,
            "command_index": 1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    base_roll_l2 = RewTerm(
        func=mdp.base_roll_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_yaw_l2 = RewTerm(
        func=mdp.base_yaw_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Jump task rewards
    jump_prep = RewTerm(
        func=mdp.jump_crouch_prep_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "target_crouch_z": 0.28,
            "std": 0.06,
            "min_contact_feet": 2,
            "threshold": 1.0,
        },
    )

    jump_clearance = RewTerm(
        func=mdp.jump_obstacle_clearance_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "body_names": ("FL_foot", "FR_foot", "RL_foot", "RR_foot", "base_link"),
            "top_height": 0.15,
            "margin": 0.03,
            "std": 0.05,
            "threshold": 1.0,
        },
    )

    landing_pred_xy = RewTerm(
        func=mdp.landing_prediction_xy_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "target_x": 1.0,
            "target_y": 0.0,
            "target_z": 0.15,
            "std_xy": 0.18,
            "threshold": 1.0,
            "gravity": 9.81,
        },
    )

    all_feet_air = RewTerm(
        func=mdp.all_feet_air,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 1.0,
        },
    )

    root_x_progress = RewTerm(
        func=mdp.root_x_progress,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_x": 0.35,
        },
    )

    touchdown_target_x = RewTerm(
        func=mdp.touchdown_target_x_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "target_x": 0.35,
            "std": 0.08,
            "min_air_time": 0.10,
        },
    )

    landing_stable = RewTerm(
        func=mdp.landing_stable_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "min_air_time": 0.10,
            "lin_vel_z_std": 0.7,
            "ang_vel_xy_std": 4.0,
            "ori_std": 0.4,
        },
    )

    block_top_pose = RewTerm(
        func=mdp.block_top_pose_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_x": 1.0,
            "target_z": 0.5,
            "x_std": 0.20,
            "z_std": 0.12,
        },
    )

    jump_success = RewTerm(
        func=mdp.jump_success_hold_bonus,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "foot_body_names": ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
            "top_center_x": 1.0,
            "top_center_y": 0.0,
            "top_height": 0.15,
            "top_half_x": 0.075,
            "top_half_y": 0.40,
            "edge_margin": 0.02,
            "hold_time_s": 0.20,
            "lin_vel_threshold": 0.25,
            "ang_vel_threshold": 1.5,
            "z_tol": 0.02,
            "contact_threshold": 1.0,
        },
    )

    jump_fail = RewTerm(
        func=mdp.jump_fail_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["^(?!.*.*_foot).*"]),
            "support_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ),
            "foot_body_names": ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
            "top_center_x": 1.0,
            "top_center_y": 0.0,
            "top_height": 0.15,
            "top_half_x": 0.075,
            "top_half_y": 0.40,
            "edge_margin": 0.02,
            "min_base_height": 0.12,
            "orientation_limit": 0.8,
            "z_tol": 0.02,
            "contact_threshold": 1.0,
            "body_contact_threshold": 1.0,
        },
    )

    aerial_tuck = RewTerm(
        func=mdp.aerial_tuck_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "front_body_names": ("FL_foot", "FR_foot"),
            "hind_body_names": ("RL_foot", "RR_foot"),
            "threshold": 1.0,
            "target_dist": 0.06,
            "std": 0.06,
        },
    )

    # Others
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=0.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
    #     },
    # )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_including_ang_z,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )

    feet_air_time_variance = RewTerm(
        func=mdp.feet_air_time_variance_penalty,
        weight=0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="")},
    )

    feet_gait = RewTerm(
        func=mdp.GaitReward,
        weight=0.0,
        params={
            "std": math.sqrt(0.5),
            "command_name": "base_velocity",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("", ""), ("", "")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    feet_contact = RewTerm(
        func=mdp.feet_contact,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "expect_contact_num": 2,
        },
    )
    feet_contact_count_penalty = RewTerm(
        func=mdp.feet_contact_count_penalty,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "expect_contact_num": 2,
            "threshold": 1.0,
            "command_threshold": 0.1,
        },
    )

    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
        },
    )

    feet_contact_ratio = RewTerm(
        func=mdp.feet_contact_ratio,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 1.0,
        },
    )
    feet_stuck_time_penalty = RewTerm(
        func=mdp.feet_stuck_time_penalty,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "max_mode_time": 0.35,
            "max_excess_time": 0.5,
            "command_threshold": 0.1,
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
        },
    )

    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
        },
    )  # negetive

    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "tanh_mult": 2.0,
            "target_height": 0.05,
            "command_name": "base_velocity",
        },
    )

    feet_height_body = RewTerm(
        func=mdp.feet_height_body,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "tanh_mult": 2.0,
            "target_height": -0.3,
            "command_name": "base_velocity",
        },
    )

    feet_distance_y_exp = RewTerm(
        func=mdp.feet_distance_y_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "stance_width": float,
        },
    )

    # feet_distance_xy_exp = RewTerm(
    #     func=mdp.feet_distance_xy_exp,
    #     weight=0.0,
    #     params={
    #         "std": math.sqrt(0.25),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=""),
    #         "stance_length": float,
    #         "stance_width": float,
    #     },
    # )

    upward = RewTerm(func=mdp.upward, weight=0.0)

    # lin_vel_xy_l2_with_ang_z_command = RewTerm(
    #     func=mdp.lin_vel_xy_l2_with_ang_z_command,
    #     weight=0,
    #     params={
    #         "command_name": "base_velocity",
    #         "command_threshold": 0.1,
    #     },
    # ) # negetive
