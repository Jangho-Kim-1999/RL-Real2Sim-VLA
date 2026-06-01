from isaaclab.utils import configclass

from .flat_env_cfg import MCLQuadserialFlatEnvCfg


@configclass
class MCLQuadserialStandEnvCfg(MCLQuadserialFlatEnvCfg):
    """Flat locomotion task without small-command stand-still rewards."""

    def __post_init__(self):
        super().__post_init__()

        # Keep this task identical to Flat-MCLQuad-serial, except do not reward
        # staying near the default pose when the velocity command is small.
        self.rewards.stand_still = None
        self.rewards.stand_still_without_cmd = None
        self.rewards.joint_pos_penalty = None
        self.rewards.hipx_joint_pos_penalty = None
        self.rewards.hipy_joint_pos_penalty = None
        self.rewards.knee_joint_pos_penalty = None
        self.rewards.feet_contact_without_cmd = None

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.rel_standing_envs = 0.2

        self.rewards.track_lin_vel_xy_exp.weight = 40.0
        self.rewards.track_ang_vel_z_exp.weight = 10.0

        self.rewards.feet_gait.weight = 4.0
        self.rewards.feet_contact_count_penalty.weight = -3.0
        self.rewards.feet_stuck_time_penalty.weight = -2.0
        self.rewards.feet_gait.params["command_threshold"] = -1.0
        self.rewards.feet_contact_count_penalty.params["command_threshold"] = -1.0
        self.rewards.feet_stuck_time_penalty.params["command_threshold"] = -1.0
        self.rewards.feet_air_time.params["command_threshold"] = -1.0
