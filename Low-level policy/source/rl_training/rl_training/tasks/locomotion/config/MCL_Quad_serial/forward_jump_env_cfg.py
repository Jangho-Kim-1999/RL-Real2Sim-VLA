# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from rl_training.tasks.locomotion.M_rewards import RewardsCfg
import rl_training.tasks.locomotion.mdp as mdp
from .flat_env_cfg import MCLQuadserialFlatEnvCfg


@configclass
class MCLQuadserialForwardJumpEnvCfg(MCLQuadserialFlatEnvCfg):
    """Flat forward jump task aligned with SPI-Active's go2_block_jump setup."""

    target_forward_distance_m = 1.0
    target_z_m = 0.4
    base_height_after_jump_m = 0.35
    reward_std = 0.25
    jump_cycle_time_s = 3.0

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Episode / Commands------------------------------
        # SPI-Active go2_block_jump uses one phase cycle across a 3 s episode.
        self.episode_length_s = float(self.jump_cycle_time_s)
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.jump_mode = None

        # ------------------------------Observations------------------------------
        # SPI-Active actor obs: base_ang_vel, projected_gravity, dof_pos, dof_vel, actions, last_actions, phase.
        self.observations.policy.enable_corruption = False
        self.observations.policy.base_lin_vel = None  # type: ignore
        self.observations.policy.height_scan = None  # type: ignore
        self.observations.policy.velocity_commands = None  # type: ignore
        self.observations.policy.actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        self.observations.policy.last_actions = ObsTerm(func=mdp.prev_action, clip=(-100.0, 100.0), scale=1.0)
        self.observations.policy.phase = ObsTerm(
            func=mdp.phase,
            params={"cycle_time": float(self.jump_cycle_time_s)},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        # SPI-Active critic additionally observes base_height and base_lin_vel.
        self.observations.critic.velocity_commands = None  # type: ignore
        self.observations.critic.height_scan = None  # type: ignore
        self.observations.critic.base_height = ObsTerm(func=mdp.base_height, clip=(-100.0, 100.0), scale=1.0)
        self.observations.critic.actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        self.observations.critic.last_actions = ObsTerm(func=mdp.prev_action, clip=(-100.0, 100.0), scale=1.0)
        self.observations.critic.phase = ObsTerm(
            func=mdp.phase,
            params={"cycle_time": float(self.jump_cycle_time_s)},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        # ------------------------------Reset / Randomization------------------------------
        # SPI-Active initializes x/y/yaw at the origin and randomizes root height in a narrow band.
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (-0.05, 0.05),
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
        # self.events.randomize_rigid_body_material = None
        # self.events.randomize_terrain_material = None
        # self.events.randomize_rigid_body_mass = None
        # self.events.randomize_rigid_body_mass_base = None
        # self.events.randomize_rigid_body_inertia = None
        # self.events.randomize_com_positions = None
        # self.events.randomize_apply_external_force_torque = None
        # self.events.randomize_push_robot = None

        # ------------------------------Task Rewards------------------------------
        self.rewards.reach_x_target = RewTerm(
            func=mdp.spi_jump_reach_x_target,
            weight=4.3,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_x": float(self.target_forward_distance_m),
                "std": float(self.reward_std),
            },
        )
        self.rewards.reach_z_target = RewTerm(
            func=mdp.spi_jump_reach_z_target,
            weight=-1.7,
            params={"asset_cfg": SceneEntityCfg("robot"), "target_z": float(self.target_z_m)},
        )
        self.rewards.feet_height_before_jump = RewTerm(
            func=mdp.spi_jump_feet_height_before_jump,
            weight=3,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
                "std": float(self.reward_std),
            },
        )
        self.rewards.actions_symmetry = RewTerm(
            func=mdp.spi_jump_actions_symmetry,
            weight=6,
            params={"std": float(self.reward_std)},
        )
        self.rewards.lin_vel_z = RewTerm(func=mdp.spi_jump_lin_vel_z, weight=3.5, params={"asset_cfg": SceneEntityCfg("robot")})
        self.rewards.lin_vel_x = RewTerm(func=mdp.spi_jump_lin_vel_x, weight=4, params={"asset_cfg": SceneEntityCfg("robot")})
        self.rewards.height_control = RewTerm(
            func=mdp.spi_jump_height_control,
            weight=-14.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "target_height": float(self.base_height_after_jump_m)},
        )
        self.rewards.penalty_orientation = RewTerm(
            func=mdp.spi_jump_penalty_orientation,
            weight=-2.2,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.penalty_slippage = RewTerm(
            func=mdp.spi_jump_penalty_slippage,
            weight=-3.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            },
        )
        self.rewards.penalty_contact_during_air = RewTerm(
            func=mdp.spi_jump_penalty_contact_during_air,
            weight=-3.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name])},
        )
        self.rewards.penalty_contact_landing = RewTerm(
            func=mdp.spi_jump_penalty_contact_landing,
            weight=-0.07,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name])},
        )
        self.rewards.feet_distance = RewTerm(
            func=mdp.spi_jump_feet_distance,
            weight=-1.5,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"])},
        )
        self.rewards.feet_x = RewTerm(
            func=mdp.spi_jump_feet_x,
            weight=-0.3,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
                "extra_stance_length": 0.15,
            },
        )

        # ------------------------------Disable Locomotion / Legacy Jump Rewards------------------------------
        for term_name in (
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "feet_gait",
            "feet_air_time",
            "feet_air_time_variance",
            "feet_contact_count_penalty",
            "feet_stuck_time_penalty",
            "stand_still",
            "base_height_l2",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "flat_orientation_l2",
            "feet_height",
            "feet_height_body",
            "inter_diagonal_load_balance",
            "diagonal_grf_balance",
            "contact_forces",
            "jump_prep",
            "all_feet_air",
            "root_x_progress",
            "touchdown_target_x",
            "landing_stable",
            "jump_lateral_drift",
            "jump_apex_height",
        ):
            term = getattr(self.rewards, term_name, None)
            if term is not None:
                term.weight = 0.0

        # ------------------------------Regularization / Local Safety------------------------------
        term = getattr(self.rewards, "action_rate_l2", None)
        if term is not None:
            term.weight = -1.0e-3
        for term_name in (
            "feet_slide",
            "undesired_contacts",
            "joint_torques_no_comp_l2",
            "joint_torques_l2",
            "joint_power_no_comp",
            "joint_power",
            "joint_acc_l2",
            "joint_vel_l2",
            "joint_pos_limits",
            "joint_deviation_l1",
            "joint_mirror",
            "feet_contact_without_cmd",
        ):
            term = getattr(self.rewards, term_name, None)
            if term is not None:
                term.weight = 0.0

        # Yaw suppression: penalize yaw-rate during jump.
        reward_defaults = RewardsCfg()
        if getattr(self.rewards, "ang_vel_z_l2", None) is None and hasattr(reward_defaults, "ang_vel_z_l2"):
            self.rewards.ang_vel_z_l2 = reward_defaults.ang_vel_z_l2
        if getattr(self.rewards, "ang_vel_z_l2", None) is not None:
            self.rewards.ang_vel_z_l2.weight = -0.05

        # Front/Rear gait synchronization reward.
        # if getattr(self.rewards, "feet_gait", None) is None and hasattr(reward_defaults, "feet_gait"):
        #     self.rewards.feet_gait = reward_defaults.feet_gait
        # if getattr(self.rewards, "feet_gait", None) is not None:
        #     self.rewards.feet_gait.weight = 0.1
        #     self.rewards.feet_gait.params["synced_feet_pair_names"] = [
        #         ["FL_foot", "FR_foot"],
        #         ["RL_foot", "RR_foot"],
        #     ]

        # SPI-Active disables gravity-orientation termination for this jump task.
        self.terminations.bad_orientation_2 = None
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_link_name]), "threshold": 1.0},
        )

        self.disable_zero_weight_rewards()
