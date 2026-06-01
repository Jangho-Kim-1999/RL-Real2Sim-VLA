# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
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
class MCLQuadserialBlockEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Flat block-climb task.

    - All envs start with the same base orientation and velocity.
    - Block front-face distance from robot is set by ``block_nearest_x_m``.
    """

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

    block_x_length_m = 1
    block_height_start_m = 0.15
    block_size_step_m = 0.15
    block_y_length_m = 0.80
    block_num_levels = 6
    block_nearest_x_m = 0.6
    block_center_x_m = 0.0  # derived from nearest distance + half x-length
    block_curriculum_switch_steps = 20_000_000

    def _block_center_x_from_length(self, x_length_m: float) -> float:
        """Compute block center-x from desired nearest front-face distance."""
        return float(self.block_nearest_x_m) + 0.5 * float(x_length_m)

    def _ensure_block_jump_terms(self) -> None:
        if getattr(self.rewards, "jump_prep", None) is None:
            self.rewards.jump_prep = RewTerm(
                func=mdp.jump_crouch_prep_exp,
                weight=0.0,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "sensor_cfg": SceneEntityCfg(
                        "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
                    ),
                    "target_crouch_z": 0.28,
                    "std": 0.06,
                    "min_contact_feet": 2,
                    "threshold": 1.0,
                },
            )
        if getattr(self.rewards, "jump_clearance", None) is None:
            self.rewards.jump_clearance = RewTerm(
                func=mdp.jump_obstacle_clearance_exp,
                weight=0.0,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "sensor_cfg": SceneEntityCfg(
                        "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
                    ),
                    "body_names": ("FL_foot", "FR_foot", "RL_foot", "RR_foot", "base_link"),
                    "top_height": float(self.block_height_start_m),
                    "margin": 0.03,
                    "std": 0.05,
                    "threshold": 1.0,
                },
            )
        if getattr(self.rewards, "landing_pred_xy", None) is None:
            self.rewards.landing_pred_xy = RewTerm(
                func=mdp.landing_prediction_xy_exp,
                weight=0.0,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "sensor_cfg": SceneEntityCfg(
                        "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
                    ),
                    "target_x": float(self.block_center_x_m),
                    "target_y": 0.0,
                    "target_z": float(self.block_height_start_m),
                    "std_xy": 0.18,
                    "threshold": 1.0,
                    "gravity": 9.81,
                },
            )
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
                    "target_x": float(self.block_center_x_m),
                },
            )
        if getattr(self.rewards, "touchdown_target_x", None) is None:
            self.rewards.touchdown_target_x = RewTerm(
                func=mdp.touchdown_target_x_exp,
                weight=0.0,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
                    "target_x": float(self.block_center_x_m),
                    "std": 0.10,
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
        if getattr(self.rewards, "block_top_pose", None) is None:
            self.rewards.block_top_pose = RewTerm(
                func=mdp.block_top_pose_exp,
                weight=0.0,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "target_x": float(self.block_center_x_m),
                    "target_z": float(self.block_height_start_m * 0.5 + 0.35),
                    "x_std": 0.20,
                    "z_std": 0.12,
                },
            )
        if getattr(self.rewards, "jump_success", None) is None:
            self.rewards.jump_success = RewTerm(
                func=mdp.jump_success_hold_bonus,
                weight=0.0,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "sensor_cfg": SceneEntityCfg(
                        "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
                    ),
                    "foot_body_names": ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
                    "top_center_x": float(self.block_center_x_m),
                    "top_center_y": 0.0,
                    "top_height": float(self.block_height_start_m),
                    "top_half_x": float(0.5 * self.block_x_length_m),
                    "top_half_y": float(0.5 * self.block_y_length_m),
                    "edge_margin": 0.02,
                    "hold_time_s": 0.20,
                    "lin_vel_threshold": 0.25,
                    "ang_vel_threshold": 1.5,
                    "z_tol": 0.02,
                    "contact_threshold": 1.0,
                },
            )
        if getattr(self.rewards, "jump_fail", None) is None:
            self.rewards.jump_fail = RewTerm(
                func=mdp.jump_fail_penalty,
                weight=0.0,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{self.foot_link_name}).*"]),
                    "support_sensor_cfg": SceneEntityCfg(
                        "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
                    ),
                    "foot_body_names": ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
                    "top_center_x": float(self.block_center_x_m),
                    "top_center_y": 0.0,
                    "top_height": float(self.block_height_start_m),
                    "top_half_x": float(0.5 * self.block_x_length_m),
                    "top_half_y": float(0.5 * self.block_y_length_m),
                    "edge_margin": 0.02,
                    "min_base_height": 0.12,
                    "orientation_limit": 0.8,
                    "z_tol": 0.02,
                    "contact_threshold": 1.0,
                    "body_contact_threshold": 1.0,
                },
            )

    def __post_init__(self):
        super().__post_init__()
        # Derive active block center from nearest-face distance and x-length.
        self.block_center_x_m = self._block_center_x_from_length(self.block_x_length_m)

        # ------------------------------Scene------------------------------
        self.scene.robot = MCLQUAD_SERIAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner.pattern_cfg.resolution = 0.07
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.ground_reaction_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.foot_link_name,
            history_length=3,
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
        # Block task command observation: [dx_to_block, dz_to_top, jump_flag].
        self.observations.policy.velocity_commands = ObsTerm(
            func=mdp.block_task_command,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "jump_mode",
                "target_x": float(self.block_center_x_m),
                "target_z": float(self.block_height_start_m),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.critic.velocity_commands = ObsTerm(
            func=mdp.block_task_command,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "jump_mode",
                "target_x": float(self.block_center_x_m),
                "target_z": float(self.block_height_start_m),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )

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

        # ------------------------------Rewards (flat baseline)------------------------------
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

        # ------------------------------Terminations/Curriculums/Commands------------------------------
        self.terminations.illegal_contact = None
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # Deterministic reset so every robot starts identically.
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

        # Keep physics fixed for this task setup.
        self.events.randomize_rigid_body_material = None
        self.events.randomize_terrain_material = None
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_inertia = None
        self.events.randomize_com_positions = None

        # Jump task uses target rewards, not velocity command tracking.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0
        # Jump mode command:
        # 0 -> stand objective, 1 -> jump objective (sampled per episode).
        self.commands.jump_mode = mdp.DiscreteCommandControllerCfg(
            resampling_time_range=(float(self.episode_length_s), float(self.episode_length_s)),
            available_commands=[0, 1],
            debug_vis=False,
        )

        # Multi-level obstacle curriculum:
        # x-length is fixed, only z-height increases by block_size_step_m.
        obstacle_names: list[str] = []
        obstacle_half_heights: list[float] = []
        obstacle_half_lengths: list[float] = []
        for level in range(int(self.block_num_levels)):
            size_x = float(self.block_x_length_m)
            size_z = float(self.block_height_start_m + level * self.block_size_step_m)
            obstacle_name = f"block_obstacle_lvl{level}"
            obstacle_names.append(obstacle_name)
            obstacle_half_heights.append(0.5 * size_z)
            obstacle_half_lengths.append(0.5 * size_x)

            is_level0 = level == 0
            init_x = self._block_center_x_from_length(size_x) if is_level0 else (100.0 + 2.0 * level)
            init_z = 0.5 * size_z if is_level0 else -10.0
            color_scale = 0.35 + 0.06 * level
            color = (min(color_scale, 0.85), min(color_scale, 0.85), min(0.45 + 0.03 * level, 0.9))

            setattr(
                self.scene,
                obstacle_name,
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{obstacle_name}",
                    spawn=sim_utils.CuboidCfg(
                        size=(size_x, float(self.block_y_length_m), size_z),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=True,
                            disable_gravity=True,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(init_x, 0.0, init_z)),
                ),
            )

        self.curriculum.command_levels = CurrTerm(
            func=mdp.block_obstacle_curriculum,
            params={
                "obstacle_names": tuple(obstacle_names),
                "obstacle_half_heights": tuple(obstacle_half_heights),
                "obstacle_half_lengths": tuple(obstacle_half_lengths),
                "switch_steps": int(self.block_curriculum_switch_steps),
                "active_x": float(self.block_center_x_m),
                "active_y": 0.0,
                "hidden_x": 100.0,
                "hidden_z": -10.0,
                "base_clearance": 0.35,
                "root_term_name": "root_x_progress",
                "touchdown_term_name": "touchdown_target_x",
                "top_pose_term_name": "block_top_pose",
                "clearance_term_name": "jump_clearance",
                "landing_pred_term_name": "landing_pred_xy",
                "success_term_name": "jump_success",
                "fail_term_name": "jump_fail",
            },
        )

        # Switch block task to jump-and-land objective.
        self._ensure_block_jump_terms()
        for term_name in (
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "feet_gait",
            "feet_air_time",
            "feet_air_time_variance",
            "stand_still",
            "base_height_l2",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "flat_orientation_l2",
            "feet_height",
            "feet_height_body",
            "joint_mirror",
        ):
            term = getattr(self.rewards, term_name, None)
            if term is not None:
                term.weight = 0.0

        self.rewards.root_x_progress.weight = 0.4
        self.rewards.root_x_progress.params["target_x"] = float(self.block_center_x_m)
        self.rewards.root_x_progress.params["command_name"] = "jump_mode"
        self.rewards.root_x_progress.params["active_when_jump"] = True
        self.rewards.jump_prep.weight = 0.4
        self.rewards.jump_prep.params["sensor_cfg"].body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.rewards.jump_prep.params["target_crouch_z"] = 0.28
        self.rewards.jump_prep.params["command_name"] = "jump_mode"
        self.rewards.jump_prep.params["active_when_jump"] = True
        self.rewards.jump_clearance.weight = 1.5
        self.rewards.jump_clearance.params["sensor_cfg"].body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.rewards.jump_clearance.params["body_names"] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot", "base_link")
        self.rewards.jump_clearance.params["top_height"] = float(self.block_height_start_m)
        self.rewards.jump_clearance.params["margin"] = 0.03
        self.rewards.jump_clearance.params["std"] = 0.05
        self.rewards.jump_clearance.params["command_name"] = "jump_mode"
        self.rewards.jump_clearance.params["active_when_jump"] = True
        self.rewards.landing_pred_xy.weight = 1.2
        self.rewards.landing_pred_xy.params["sensor_cfg"].body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.rewards.landing_pred_xy.params["target_x"] = float(self.block_center_x_m)
        self.rewards.landing_pred_xy.params["target_y"] = 0.0
        self.rewards.landing_pred_xy.params["target_z"] = float(self.block_height_start_m)
        self.rewards.landing_pred_xy.params["std_xy"] = 0.18
        self.rewards.landing_pred_xy.params["command_name"] = "jump_mode"
        self.rewards.landing_pred_xy.params["active_when_jump"] = True
        self.rewards.all_feet_air.weight = 0.8
        self.rewards.all_feet_air.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.all_feet_air.params["command_name"] = "jump_mode"
        self.rewards.all_feet_air.params["active_when_jump"] = True
        self.rewards.touchdown_target_x.weight = 8.0
        self.rewards.touchdown_target_x.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.touchdown_target_x.params["target_x"] = float(self.block_center_x_m)
        self.rewards.touchdown_target_x.params["std"] = 0.10
        self.rewards.touchdown_target_x.params["min_air_time"] = 0.10
        self.rewards.touchdown_target_x.params["command_name"] = "jump_mode"
        self.rewards.touchdown_target_x.params["active_when_jump"] = True
        self.rewards.landing_stable.weight = 3.0
        self.rewards.landing_stable.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.landing_stable.params["command_name"] = "jump_mode"
        self.rewards.landing_stable.params["active_when_jump"] = True
        self.rewards.block_top_pose.weight = 6.0
        self.rewards.block_top_pose.params["target_x"] = float(self.block_center_x_m)
        self.rewards.block_top_pose.params["target_z"] = float(self.block_height_start_m * 0.5 + 0.35)
        self.rewards.block_top_pose.params["x_std"] = 0.20
        self.rewards.block_top_pose.params["z_std"] = 0.12
        self.rewards.block_top_pose.params["command_name"] = "jump_mode"
        self.rewards.block_top_pose.params["active_when_jump"] = True
        self.rewards.jump_success.weight = 12.0
        self.rewards.jump_success.params["sensor_cfg"].body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.rewards.jump_success.params["top_center_x"] = float(self.block_center_x_m)
        self.rewards.jump_success.params["top_center_y"] = 0.0
        self.rewards.jump_success.params["top_height"] = float(self.block_height_start_m)
        self.rewards.jump_success.params["top_half_x"] = float(0.5 * self.block_x_length_m)
        self.rewards.jump_success.params["top_half_y"] = float(0.5 * self.block_y_length_m)
        self.rewards.jump_success.params["command_name"] = "jump_mode"
        self.rewards.jump_success.params["active_when_jump"] = True
        self.rewards.jump_success.params["switch_to_stand_after_success"] = True
        self.rewards.jump_success.params["switch_command_name"] = "jump_mode"
        self.rewards.jump_success.params["stand_value"] = 0
        self.rewards.jump_fail.weight = -4.0
        self.rewards.jump_fail.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.jump_fail.params["support_sensor_cfg"].body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.rewards.jump_fail.params["top_center_x"] = float(self.block_center_x_m)
        self.rewards.jump_fail.params["top_center_y"] = 0.0
        self.rewards.jump_fail.params["top_height"] = float(self.block_height_start_m)
        self.rewards.jump_fail.params["top_half_x"] = float(0.5 * self.block_x_length_m)
        self.rewards.jump_fail.params["top_half_y"] = float(0.5 * self.block_y_length_m)
        self.rewards.jump_fail.params["command_name"] = "jump_mode"
        self.rewards.jump_fail.params["active_when_jump"] = True

        # Stand objective when jump_flag == 0.
        self.rewards.stand_still.weight = -1.0
        self.rewards.stand_still.params["command_name"] = "jump_mode"
        self.rewards.stand_still.params["command_threshold"] = 0.5
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names

        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.joint_torques_no_comp_l2.weight = -2e-5
        self.rewards.joint_torques_l2.weight = 0.0
        self.rewards.joint_power_no_comp.weight = -1e-5
        self.rewards.joint_power.weight = 0.0
        self.rewards.joint_acc_l2.weight = -1e-7
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.contact_forces.weight = -5e-3
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.joint_deviation_l1.weight = -0.2
        self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = [".*HAA.*"]

        self.disable_zero_weight_rewards()
