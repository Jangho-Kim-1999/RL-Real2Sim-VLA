# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
import rl_training.tasks.locomotion.mdp as mdp
from rl_training.assets.MCLrobotics import (  # isort: skip
    Bm,
    Jm,
    MCLQUAD_SERIAL_CFG,
    MCLQUAD_SERIAL_TORQUE_DELAY_ENABLE,
    MCLQUAD_SERIAL_TORQUE_DELAY_MAX_STEPS,
    MCLQUAD_SERIAL_TORQUE_DELAY_MIN_STEPS,
)
from rl_training.tasks.locomotion.Env import LocomotionVelocityRoughEnvCfg
@configclass
class MCLQuadserialFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Flat terrain env configured independently from rough env."""
    base_link_name = "base_link"
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        "FLHAA", "FLHIP", "FLKNEE",
        "FRHAA", "FRHIP", "FRKNEE",
        "RLHAA", "RLHIP", "RLKNEE",
        "RRHAA", "RRHIP", "RRKNEE",
    ]
    whole_link_names = [
       "base_link",
       "FL_torso", "FR_torso", "RL_torso", "RR_torso",
       "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
       "FL_shank", "FR_shank", "RL_shank", "RR_shank",
       "FL_foot", "FR_foot", "RL_foot", "RR_foot",
    ]
    link_names = [
       "FL_torso", "FR_torso", "RL_torso", "RR_torso",
       "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
       "FL_shank", "FR_shank", "RL_shank", "RR_shank",
       "FL_foot", "FR_foot", "RL_foot", "RR_foot",
    ]
    # fmt: on
    def __post_init__(self):
        super().__post_init__()
        # ------------------------------Scene------------------------------
        self.scene.robot = MCLQUAD_SERIAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner.pattern_cfg.resolution = 0.07
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.ground_reaction_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.foot_link_name,
            history_length=3,
            force_threshold=0.0,
            debug_vis=False,
            filter_prim_paths_expr=[self.scene.terrain.prim_path + "/terrain/GroundPlane/CollisionPlane"],
        )
        self.scene.ground_reaction_forces.update_period = self.sim.dt
        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel = None  # type: ignore
        self.observations.policy.height_scan = None  # type: ignore
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        # ------------------------------Actions------------------------------
        self.actions.joint_pos.scale = {".*HAA": 0.125, "^(?!.*HAA).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names
        # ------------------------------Motor delay------------------------------
        self.torque_delay_enable = bool(MCLQUAD_SERIAL_TORQUE_DELAY_ENABLE)
        self.torque_delay_min_steps = max(0, int(MCLQUAD_SERIAL_TORQUE_DELAY_MIN_STEPS))
        self.torque_delay_max_steps = max(self.torque_delay_min_steps, int(MCLQUAD_SERIAL_TORQUE_DELAY_MAX_STEPS))
        self.torque_delay_steps = self.torque_delay_max_steps
        self.Jm = float(Jm)
        self.Bm = float(Bm)
        self.pd_gain_randomization_enable = True
        self.pd_kp_range = (50, 50)
        self.pd_kd_range = (1.5, 1.5)
        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.func = mdp.reset_root_state_uniform_with_foot_projection
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (-0.0, 0.0),
            },
            "foot_body_names": self.foot_link_name,
            "min_foot_clearance": 0.01,
        }
    #! Whole links
        self.events.randomize_rigid_body_material.params["asset_cfg"].body_names = self.whole_link_names
        # self.events.randomize_rigid_body_material = None
    #! Body
        # self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = self.base_link_name
        self.events.randomize_rigid_body_mass_base = None
        #! Whole & Leg
        # self.events.randomize_com_positions.params["asset_cfg"].body_names = self.base_link_name
        self.events.randomize_com_positions.params["asset_cfg"].body_names = self.whole_link_names
        # self.events.randomize_com_positions = None
        #! Whole & Leg
        # self.events.randomize_rigid_body_inertia.params["asset_cfg"].body_names = self.base_link_name
        self.events.randomize_rigid_body_inertia.params["asset_cfg"].body_names = self.whole_link_names
        # self.events.randomize_rigid_body_inertia = None
        #? Leg
        # self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = self.link_names
        self.events.randomize_rigid_body_mass = None
    #? External Force
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
# ------------------------------Rewards------------------------------
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.base_height_l2.weight = -8.0
        self.rewards.base_height_l2.params["target_height"] = 0.35
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.feet_air_time.weight = 2.5
        self.rewards.feet_air_time.params["threshold"] = 0.3
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -7.5
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -1.81
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.stand_still.params["command_threshold"] = 0.1
        self.rewards.feet_height_body.weight = -1.5
        self.rewards.feet_height_body.params["target_height"] = -0.35
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = -0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.contact_forces.weight = -1.0e-2
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.contact_forces.params["threshold"] = 300.0
        self.rewards.diagonal_grf_balance.weight = 0.0
        self.rewards.inter_diagonal_load_balance.weight = -0.3
        self.rewards.lin_vel_z_l2.weight = -3.8
        self.rewards.ang_vel_xy_l2.weight = -2.0
        self.rewards.track_lin_vel_xy_exp.weight = 9.2
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.joint_torques_no_comp_l2.weight = -2.0e-6
        self.rewards.joint_acc_l2.weight = -1e-6
        self.rewards.joint_deviation_l1.weight = -1.0
        self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = [".*HAA.*"]
        self.rewards.joint_power_no_comp.weight = -2e-5
        self.rewards.applied_torque_limits_no_comp.weight = -1e-4
        self.rewards.applied_torque_limits_no_comp.params["torque_limit"] = 60.0
        self.rewards.flat_orientation_l2.weight = -7.0
        self.rewards.joint_power.weight = 0.0
        self.rewards.applied_torque_limits.weight = 0.0
        self.rewards.joint_torques_l2.weight = 0.0
        self.rewards.feet_gait.weight = 2.5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_foot", "RR_foot"],
            ["FR_foot", "RL_foot"],
        ]
        self.rewards.feet_contact_count_penalty.weight = -1.5
        self.rewards.feet_contact_count_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stuck_time_penalty.weight = -1.0
        self.rewards.feet_stuck_time_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.joint_mirror.weight = -0.15
        self.rewards.joint_mirror.params["mirror_joints"] = [
        ["FL(HAA|HIP|KNEE).*", "RR(HAA|HIP|KNEE).*"],
        ["FR(HAA|HIP|KNEE).*", "RL(HAA|HIP|KNEE).*"],
        ]
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["soft_ratio"] = 1.0
        self.rewards.joint_pos_limits.params["margin_ratio"] = 0.15
        self.rewards.joint_pos_limits.params["power"] = 2.0
        self.rewards.joint_pos_limits.params["limit_overrides"] = None
        self.rewards.joint_pos_limits.params["bispace_limit_overrides"] = {
            "QM": (0.0, 3.0),
            "QB": (1.0, 3.0),
        }
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        # remove zero-weight placeholders so empty body regex terms are never parsed
        self.disable_zero_weight_rewards()
        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None
        # self.terminations.joint_rom_violation = None
        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None
        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-2.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
