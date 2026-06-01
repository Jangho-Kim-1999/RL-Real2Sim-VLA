# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

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
class MCLQuadserialAgileJumpEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Gap-crossing task on a dedicated agile-jump terrain."""

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

    def _ensure_jump_reward_terms(self) -> None:
        if getattr(self.rewards, "all_feet_air", None) is None:
            self.rewards.all_feet_air = RewTerm(
                func=mdp.all_feet_air,
                weight=0.0,
                params={
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
                    "threshold": 1.0,
                },
            )
        if getattr(self.rewards, "root_x_progress", None) is None:
            self.rewards.root_x_progress = RewTerm(
                func=mdp.root_x_progress,
                weight=0.0,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "target_x": 0.35,
                },
            )
        if getattr(self.rewards, "touchdown_target_x", None) is None:
            self.rewards.touchdown_target_x = RewTerm(
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
        if getattr(self.rewards, "landing_stable", None) is None:
            self.rewards.landing_stable = RewTerm(
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
        if getattr(self.rewards, "aerial_tuck", None) is None:
            self.rewards.aerial_tuck = RewTerm(
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

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Flat baseline (independent)------------------------------
        self.scene.robot = MCLQUAD_SERIAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner.pattern_cfg.resolution = 0.07
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.observations.policy.base_lin_vel = None  # type: ignore
        self.observations.policy.height_scan = None  # type: ignore
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        self.actions.joint_pos.scale = {".*HAA": 0.125, "^(?!.*HAA).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        self.torque_delay_enable = bool(MCLQUAD_SERIAL_TORQUE_DELAY_ENABLE)
        self.torque_delay_min_steps = max(0, int(MCLQUAD_SERIAL_TORQUE_DELAY_MIN_STEPS))
        self.torque_delay_max_steps = max(self.torque_delay_min_steps, int(MCLQUAD_SERIAL_TORQUE_DELAY_MAX_STEPS))
        self.torque_delay_steps = self.torque_delay_max_steps
        self.Jm = float(Jm)
        self.Bm = float(Bm)
        self.pd_gain_randomization_enable = True
        self.pd_kp_range = (50, 50)
        self.pd_kd_range = (1.5, 1.5)

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

        self.events.randomize_rigid_body_material.params["asset_cfg"].body_names = self.whole_link_names
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = self.base_link_name
        self.events.randomize_com_positions.params["asset_cfg"].body_names = self.base_link_name
        self.events.randomize_rigid_body_inertia.params["asset_cfg"].body_names = self.base_link_name
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = self.base_link_name
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        self.rewards.action_rate_l2.weight = -0.04
        self.rewards.base_height_l2.weight = -8.0
        self.rewards.base_height_l2.params["target_height"] = 0.35
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.feet_air_time.weight = 5.0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -8.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.5
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.stand_still.params["command_threshold"] = 0.1
        self.rewards.feet_height_body.weight = -2.5
        self.rewards.feet_height_body.params["target_height"] = -0.35
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = -0.2
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.contact_forces.weight = -2.0e-2
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.undesired_contacts.weight = -0.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.joint_torques_no_comp_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = -1e-7
        self.rewards.joint_deviation_l1.weight = -0.5
        self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = [".*HAA.*"]
        self.rewards.joint_power_no_comp.weight = -2e-5
        self.rewards.joint_power.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.feet_gait.weight = 1.5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [["FL_foot", "RR_foot"], ["FR_foot", "RL_foot"]]
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FL(HAA|HIP|KNEE).*", "RR(HAA|HIP|KNEE).*"],
            ["FR(HAA|HIP|KNEE).*", "RL(HAA|HIP|KNEE).*"],
        ]
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.terminations.illegal_contact = None
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # ------------------------------Jump-specific------------------------------
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "gap": terrain_gen.MeshGapTerrainCfg(
                    proportion=1.0,
                    platform_width=1.5,
                    gap_width_range=(0.25, 0.45),
                ),
            },
        )
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.ground_reaction_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.foot_link_name,
            history_length=3,
            debug_vis=False,
            filter_prim_paths_expr=[self.scene.terrain.prim_path + "/terrain/mesh"],
        )
        self.scene.ground_reaction_forces.update_period = self.sim.dt
        self.episode_length_s = 2.0

        self.observations.policy.base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.policy.velocity_commands = None  # type: ignore
        self.observations.critic.velocity_commands = None  # type: ignore
        self.observations.policy.height_scan = None  # type: ignore
        self.observations.critic.height_scan = None  # type: ignore

        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.randomize_reset_joints_biarticular_hip_knee.params["qm_range"] = (-0.05, 0.05)
        self.events.randomize_reset_joints_biarticular_hip_knee.params["qb_range"] = (0.05, 0.20)

        self.events.randomize_rigid_body_material = None
        self.events.randomize_terrain_material = None
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_inertia = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0

        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{self.foot_link_name}).*"]),
                "threshold": 1.0,
            },
        )

        self.actions.joint_pos.scale = {".*HAA": 0.125, "^(?!.*HAA).*": 0.30}

        self._ensure_jump_reward_terms()

        for term_name in (
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "feet_air_time",
            "feet_air_time_variance",
            "feet_gait",
            "stand_still",
            "base_height_l2",
            "lin_vel_z_l2",
            "feet_height",
            "feet_height_body",
            "ang_vel_xy_l2",
            "flat_orientation_l2",
        ):
            term = getattr(self.rewards, term_name, None)
            if term is not None:
                term.weight = 0.0

        self.rewards.root_x_progress.weight = 0.5
        self.rewards.root_x_progress.params["target_x"] = 0.35
        self.rewards.all_feet_air.weight = 1.0
        self.rewards.all_feet_air.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.touchdown_target_x.weight = 8.0
        self.rewards.touchdown_target_x.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.touchdown_target_x.params["target_x"] = 0.35
        self.rewards.touchdown_target_x.params["std"] = 0.08
        self.rewards.touchdown_target_x.params["min_air_time"] = 0.10
        self.rewards.landing_stable.weight = 3.0
        self.rewards.landing_stable.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.landing_stable.params["min_air_time"] = 0.10
        self.rewards.landing_stable.params["lin_vel_z_std"] = 0.7
        self.rewards.landing_stable.params["ang_vel_xy_std"] = 4.0
        self.rewards.landing_stable.params["ori_std"] = 0.4
        self.rewards.aerial_tuck.weight = 0.0
        self.rewards.aerial_tuck.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.joint_torques_no_comp_l2.weight = -2e-5
        self.rewards.joint_torques_l2.weight = 0.0
        self.rewards.joint_power_no_comp.weight = -1e-5
        self.rewards.joint_power.weight = 0.0
        self.rewards.joint_acc_l2.weight = -1e-7
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.contact_forces.weight = -3e-3
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.joint_deviation_l1.weight = -0.2
        self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = [".*HAA.*"]

        self.curriculum.command_levels = CurrTerm(
            func=mdp.jump_reward_weight_schedule,
            params={
                "root_term_name": "root_x_progress",
                "touchdown_term_name": "touchdown_target_x",
                "landing_term_name": "landing_stable",
                "switch_steps": 20_000_000,
                "early_root_weight": 0.5,
                "late_root_weight": 0.1,
                "early_touchdown_weight": 8.0,
                "late_touchdown_weight": 10.0,
                "early_landing_weight": 3.0,
                "late_landing_weight": 4.0,
            },
        )

        self.disable_zero_weight_rewards()
