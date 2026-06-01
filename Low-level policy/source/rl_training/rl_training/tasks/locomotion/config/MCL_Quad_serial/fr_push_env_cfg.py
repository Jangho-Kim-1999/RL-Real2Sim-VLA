# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import rl_training.tasks.locomotion.mdp as mdp
from rl_training.tasks.locomotion.config.MCL_Quad_serial.flat_env_cfg import MCLQuadserialFlatEnvCfg


@configclass
class MCLQuadserialFRPushEnvCfg(MCLQuadserialFlatEnvCfg):
    """FR-foot fixed-wall force task on flat terrain.

    The robot starts from a fixed stance. FL/RL/RR are rewarded for staying in
    ground support while only FR pushes a fixed wall in front of the body.
    """

    support_foot_names = ("FL_foot", "RL_foot", "RR_foot")
    fr_foot_name = "FR_foot"
    body_ground_termination_names = ("base_link", ".*_torso")
    body_ground_termination_threshold_n = 20.0
    body_ground_termination_grace_s = 0.25

    wall_name = "push_wall"
    wall_size = (0.04, 0.50, 0.40)
    wall_center_x = 0.40
    wall_center_x_randomization_range = (-0.08, 0.08)
    wall_center_y = -0.16
    wall_target_z = 0.12
    wall_target_force_n = 35.0
    wall_force_std_n = 15.0
    fr_wall_x_distance_std_m = 0.12
    fr_foot_target_height_m = 0.20
    fr_foot_height_std_m = 0.08

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 6.0
        self.scene.num_envs = 2048

        wall_half_x = 0.5 * float(self.wall_size[0])
        wall_half_z = 0.5 * float(self.wall_size[2])
        wall_contact_x = float(self.wall_center_x) - wall_half_x

        # ------------------------------Scene------------------------------
        setattr(
            self.scene,
            self.wall_name,
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PushWall",
                spawn=sim_utils.CuboidCfg(
                    size=tuple(float(value) for value in self.wall_size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8,
                        dynamic_friction=0.6,
                        restitution=0.0,
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.44, 0.78)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(float(self.wall_center_x), float(self.wall_center_y), wall_half_z)
                ),
            ),
        )
        setattr(
            self.scene,
            "wall_contact_forces",
            ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + self.fr_foot_name,
                history_length=3,
                force_threshold=0.0,
                debug_vis=False,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/PushWall.*"],
            ),
        )
        self.scene.wall_contact_forces.update_period = self.sim.dt
        setattr(
            self.scene,
            "body_ground_contact_forces",
            ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
                history_length=3,
                force_threshold=0.0,
                debug_vis=False,
                filter_prim_paths_expr=[self.scene.terrain.prim_path + "/terrain/GroundPlane/CollisionPlane"],
            ),
        )
        self.scene.body_ground_contact_forces.update_period = self.sim.dt

        # ------------------------------Observations------------------------------
        task_obs_params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_force_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.fr_foot_name]),
            "fr_foot_body_name": self.fr_foot_name,
            "wall_contact_x": wall_contact_x,
            "wall_cfg": SceneEntityCfg(self.wall_name),
            "wall_half_x": wall_half_x,
            "command_force": float(self.wall_target_force_n),
        }
        self.observations.policy.velocity_commands = ObsTerm(
            func=mdp.fr_wall_push_actor_task_state,
            params=task_obs_params,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.critic.velocity_commands = ObsTerm(
            func=mdp.fr_wall_push_critic_task_state,
            params={
                **task_obs_params,
                "wall_force_sensor_cfg": SceneEntityCfg("wall_contact_forces", body_names=[self.fr_foot_name]),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # ------------------------------Events------------------------------
        # Keep the task aligned with +x so the front wall is well-defined.
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
        self.events.reset_push_wall = EventTerm(
            func=mdp.reset_rigid_object_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(self.wall_name),
                "pose_range": {
                    "x": tuple(float(value) for value in self.wall_center_x_randomization_range),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )

        # Start deterministic. Add domain randomization back after the basic skill is learned.
        self.events.randomize_rigid_body_material = None
        self.events.randomize_terrain_material = None
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_inertia = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.rel_heading_envs = 0.0

        # ------------------------------Rewards------------------------------
        # Disable locomotion gait rewards that would fight the single-leg manipulation objective.
        for term_name in (
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "feet_air_time",
            "feet_air_time_variance",
            "feet_gait",
            "feet_contact_count_penalty",
            "feet_stuck_time_penalty",
            "feet_height",
            "feet_height_body",
            "stand_still",
            "joint_mirror",
            "diagonal_grf_balance",
            "inter_diagonal_load_balance",
        ):
            term = getattr(self.rewards, term_name, None)
            if term is not None:
                term.weight = 0.0

        # Preserve a quiet, upright base while the FR leg pushes the wall.
        self.rewards.base_height_l2.weight = -20.0
        self.rewards.base_height_l2.params["target_height"] = 0.35
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.root_xy_deviation = RewTerm(
            func=mdp.root_xy_deviation_l2,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "target_x": 0.0, "target_y": 0.0},
        )
        # self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.bad_orientation_2_penalty = None

        self.rewards.flat_orientation_l2.weight = -12.0
        # self.rewards.bad_orientation_2_penalty = RewTerm(
        #     func=mdp.bad_orientation_2,
        #     weight=-50.0,
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        # )
        self.rewards.lin_vel_z_l2.weight = -0.01
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.lin_vel_x_l2 = RewTerm(
            func=mdp.lin_vel_x_l2, weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        self.rewards.lin_vel_y_l2 = RewTerm(
            func=mdp.lin_vel_y_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        self.rewards.ang_vel_z_l2 = RewTerm(
            func=mdp.ang_vel_z_l2, weight=-0.03, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        # Wall-force task rewards. The wall-contact sensor is filtered to FR_foot vs PushWall only.
        # self.rewards.fr_wall_force = RewTerm(
        #     func=mdp.fr_wall_normal_force,
        #     weight=6.0,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("wall_contact_forces", body_names=[self.fr_foot_name]),
        #         "target_force": float(self.wall_target_force_n),
        #         "force_axis": 0,
        #         "use_absolute": True,
        #     },
        # )
        # self.rewards.fr_wall_force_tracking = RewTerm(
        #     func=mdp.fr_wall_command_force_tracking_exp,
        #     weight=4.0,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("wall_contact_forces", body_names=[self.fr_foot_name]),
        #         "command_force": float(self.wall_target_force_n),
        #         "force_std": float(self.wall_force_std_n),
        #         "force_axis": 0,
        #         "use_absolute": True,
        #     },
        # # )
        # self.rewards.fr_wall_tangential_force = RewTerm(
        #     func=mdp.fr_wall_tangential_force_l2,
        #     weight=-2.0e-3,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("wall_contact_forces", body_names=[self.fr_foot_name]),
        #         "normal_axis": 0,
        #     },
        # )
        self.rewards.fr_wall_contact = RewTerm(
            func=mdp.fr_wall_contact_reward,
            weight=10,
            params={
                "sensor_cfg": SceneEntityCfg("wall_contact_forces", body_names=[self.fr_foot_name]),
                "threshold": 0.5,
                "saturation_force": 0.1,
            },
        )
        self.rewards.fr_foot_body_wall_xz = RewTerm(
            func=mdp.fr_foot_body_wall_xz_exp,
            weight=15.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "foot_body_name": self.fr_foot_name,
                "wall_contact_x": wall_contact_x,
                "wall_cfg": SceneEntityCfg(self.wall_name),
                "wall_half_x": wall_half_x,
                "target_z_b": 0.3,
                "std": float(self.fr_wall_x_distance_std_m),
            },
        )
        # self.rewards.fr_wall_x_distance = RewTerm(
        #     func=mdp.fr_wall_x_distance_exp,
        #     weight=3.0,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot"),
        #         "foot_body_name": self.fr_foot_name,
        #         "wall_contact_x": wall_contact_x,
        #         "std": float(self.fr_wall_x_distance_std_m),
        #     },
        # )
        # self.rewards.fr_foot_height = RewTerm(
        #     func=mdp.fr_foot_height_exp,
        #     weight=2.0,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot"),
        #         "foot_body_name": self.fr_foot_name,
        #         "target_height": float(self.fr_foot_target_height_m),
        #         "std": float(self.fr_foot_height_std_m),
        #     },
        # )

        # Support feet should stay planted while FR alone pushes the wall.
        self.rewards.support_feet_contact = RewTerm(
            func=mdp.selected_feet_ground_contact,
            weight=10.0,
            params={
                "sensor_cfg": SceneEntityCfg("ground_reaction_forces", body_names=list(self.support_foot_names)),
                "threshold": 1.0,
            },
        )
        self.rewards.fr_ground_contact_penalty = RewTerm(
            func=mdp.selected_feet_ground_contact_penalty,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg("ground_reaction_forces", body_names=[self.fr_foot_name]),
                "threshold": 1.0,
            },
        )
        # self.rewards.support_feet_anchor = RewTerm(
        #     func=mdp.support_feet_env_position_l2,
        #     weight=-35.0,
        #     params={
        #         "asset_cfg": SceneEntityCfg(
        #             "robot", body_names=list(self.support_foot_names), preserve_order=True
        #         ),
        #         "axes": "xy",
        #     },
        # )

        self.rewards.feet_slide.weight = 0.0
        # self.rewards.feet_slide.weight = -2.0
        # self.rewards.feet_slide.params["sensor_cfg"] = SceneEntityCfg(
        #     "ground_reaction_forces", body_names=list(self.support_foot_names)
        # )
        # self.rewards.feet_slide.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", body_names=list(self.support_foot_names), preserve_order=True
        # )

        self.rewards.joint_deviation_l1.weight = -0.1
        self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = [
            "FL(HAA|HIP|KNEE).*",
            "RL(HAA|HIP|KNEE).*",
            "RR(HAA|HIP|KNEE).*",
        ]
        self.rewards.joint_torques_no_comp_l2.weight = -2.0e-6
        self.rewards.joint_power_no_comp.weight = -1.0e-5
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.action_rate_l2.weight = -0.03
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.contact_forces.weight = 0.0  # -5.0e-3
        # self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        # self.rewards.contact_forces.params["threshold"] = 350.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.body_haa_ground_contact_penalty = RewTerm(
            func=mdp.undesired_contacts,
            weight=-20.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "body_ground_contact_forces", body_names=list(self.body_ground_termination_names)
                ),
                "threshold": 1.0,
            },
        )

        # ------------------------------Terminations/Curriculums------------------------------
        self.terminations.illegal_contact = None
        self.terminations.joint_rom_violation = None
        # self.terminations.body_haa_ground_contact = DoneTerm(
        #     func=mdp.illegal_contact_after_time,
        #     params={
        #         "sensor_cfg": SceneEntityCfg(
        #             "body_ground_contact_forces", body_names=list(self.body_ground_termination_names)
        #         ),
        #         "threshold": float(self.body_ground_termination_threshold_n),
        #         "min_time_s": float(self.body_ground_termination_grace_s),
        #     },
        # )
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None

        self.disable_zero_weight_rewards()
