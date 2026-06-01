# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from rl_training.tasks.locomotion.Env import LocomotionVelocityRoughEnvCfg
import rl_training.tasks.locomotion.mdp as mdp
# from isaaclab.sensors.ray_caster import GridPatternCfg
##
# Pre-defined configs
##
# from rl_training.assets.deeprobotics import DEEPROBOTICS_LITE3_CFG  # isort: skip
from rl_training.assets.MCLrobotics import (  # isort: skip
    MCLQUAD_SERIAL_CFG,
    MCLQUAD_SERIAL_TORQUE_DELAY_ENABLE,
    MCLQUAD_SERIAL_TORQUE_DELAY_MIN_STEPS,
    MCLQUAD_SERIAL_TORQUE_DELAY_MAX_STEPS,
    Jm,
    Bm,
)


@configclass
class MCLQuadserialRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    # base_link_name = "TORSO"
    # foot_link_name = ".*_FOOT"
    # # fmt: off
    # joint_names = [
    #     "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    #     "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    #     "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    #     "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
    # ]

    # link_names = [
    #    'TORSO', 
    #    'FL_HIP', 'FR_HIP', 'HL_HIP', 'HR_HIP', 
    #    'FL_THIGH', 'FR_THIGH', 'HL_THIGH', 'HR_THIGH', 
    #    'FL_SHANK', 'FR_SHANK', 'HL_SHANK', 'HR_SHANK', 
    #    'FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT',
    # ]
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
       'base_link', 
       'FL_torso', 'FR_torso', 'RL_torso', 'RR_torso', 
       'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh', 
       'FL_shank', 'FR_shank', 'RL_shank', 'RR_shank', 
       'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot',
    ]

    link_names = [ 
       'FL_torso', 'FR_torso', 'RL_torso', 'RR_torso', 
       'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh', 
       'FL_shank', 'FR_shank', 'RL_shank', 'RR_shank', 
       'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot',
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = MCLQUAD_SERIAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner.pattern_cfg.resolution = 0.07 #  = GridPatternCfg(resolution=0.07, size=[1.6, 1.0]),
        self.scene.ground_reaction_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.foot_link_name,
            history_length=3,
            debug_vis=False,
            filter_prim_paths_expr=[self.scene.terrain.prim_path + "/terrain/mesh"],
        )
        self.scene.ground_reaction_forces.update_period = self.sim.dt

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel = None # type: ignore
        self.observations.policy.height_scan = None # type: ignore
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*HAA": 0.125, "^(?!.*HAA).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Motor delay------------------------------
        self.torque_delay_enable = bool(MCLQUAD_SERIAL_TORQUE_DELAY_ENABLE)
        self.torque_delay_min_steps = max(0, int(MCLQUAD_SERIAL_TORQUE_DELAY_MIN_STEPS))
        self.torque_delay_max_steps = max(self.torque_delay_min_steps, int(MCLQUAD_SERIAL_TORQUE_DELAY_MAX_STEPS))
        # Keep legacy field for backward compatibility in configs/log dumps.
        self.torque_delay_steps = self.torque_delay_max_steps
        self.Jm = float(Jm)
        self.Bm = float(Bm)
        self.pd_gain_randomization_enable = True
        self.pd_kp_range = (50, 50) 
        self.pd_kd_range = (1.5, 1.5)
        # self.pd_kp_range = (42.5, 57.5) # +-15%
        # self.pd_kd_range = (1.275, 1.725) # +-15%

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
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.0, 0.0),
            },
            "foot_body_names": self.foot_link_name,
            "min_foot_clearance": 0.01,
        }

    #! Whole links
        self.events.randomize_rigid_body_material.params["asset_cfg"].body_names = self.whole_link_names 
        # self.events.randomize_rigid_body_material = None


    #! Body
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = self.base_link_name 
        # self.events.randomize_rigid_body_mass_base = None
        
        self.events.randomize_com_positions.params["asset_cfg"].body_names = self.base_link_name 
        # self.events.randomize_com_positions = None
            
        self.events.randomize_rigid_body_inertia.params["asset_cfg"].body_names = self.base_link_name 
        # self.events.randomize_rigid_body_inertia = None

        #? Leg
        # self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = self.base_link_name 
        self.events.randomize_rigid_body_mass = None 

        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = self.base_link_name 
        # self.events.randomize_rigid_body_mass_base = None

    #? External Force
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        # set terrain generation probability to 0 for boxes and stairs
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.4
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.3
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.3
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        # scale down the terrains because the robot is small
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_width = 0.8
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Rewards------------------------------
        self.rewards.action_rate_l2.weight = -0.04 #-0.02
        # self.rewards.smoothness_2.weight = -0.0075

        self.rewards.base_height_l2.weight = -10.0
        self.rewards.base_height_l2.params["target_height"] = 0.35
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.feet_air_time.weight = 5.0 
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -8.0 
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.stand_still.weight = -1.2 # -1.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.stand_still.params["command_threshold"] = 0.1
        self.rewards.feet_height_body.weight = -2.5 # -2.5
        self.rewards.feet_height_body.params["target_height"] = -0.35
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = -0.2 # -0.2
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.contact_forces.weight = -2e-2 
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.lin_vel_z_l2.weight = -2.0 #-2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05 # -0.05

        self.rewards.track_lin_vel_xy_exp.weight = 3.5 #3
        self.rewards.track_ang_vel_z_exp.weight = 2.0 # 1.5

        self.rewards.undesired_contacts.weight = -0.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]

        self.rewards.joint_torques_no_comp_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = -1e-7 #-1e-7 # -1e-8 
        self.rewards.joint_deviation_l1.weight = -0.5
        self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = [".*HAA.*"]
        self.rewards.joint_power_no_comp.weight = -2e-5
        self.rewards.joint_power.weight = 0.0
        self.rewards.applied_torque_limits_no_comp.weight = -1e-4
        self.rewards.applied_torque_limits_no_comp.params["torque_limit"] = 60.0
        self.rewards.applied_torque_limits.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -5.0

        # add the following rewards to improve the gait
        self.rewards.feet_gait.weight = 2.0  # 1.5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_foot", "RR_foot"],
            ["FR_foot", "RL_foot"]
        ]

        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FL(HAA|HIP|KNEE).*", "RR(HAA|HIP|KNEE).*"],
            ["FR(HAA|HIP|KNEE).*", "RL(HAA|HIP|KNEE).*"],
        ]

        self.rewards.joint_pos_limits.weight = -5.0
        # self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]


        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "MCLQuadserialRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        # self.terminations.bad_orientation_2 = None

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # self.commands.base_velocity.ranges.lin_vel_x = (1.5, 1.5)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
