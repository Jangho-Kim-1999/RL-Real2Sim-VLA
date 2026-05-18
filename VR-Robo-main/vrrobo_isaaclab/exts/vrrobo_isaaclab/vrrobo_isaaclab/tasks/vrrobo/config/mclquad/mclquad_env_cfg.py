from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import vrrobo_isaaclab.tasks.vrrobo.mdp as mdp
import vrrobo_isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import TiledCameraCfg
from vrrobo_isaaclab.assets import MCLQUAD_SERIAL_CFG, MCLQUAD_SERIAL_JOINT_NAMES
from vrrobo_isaaclab.tasks.vrrobo.config.go2.go2_env_cfg import (
    ASSET_OFFSET,
    CommandsCfg,
    EventCfg,
    MySceneCfg,
    TerminationsCfg,
    UnitreeGo2GSBaseEnvCfg,
)


def quat_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """Convert roll/pitch/yaw degrees to a quaternion in (w, x, y, z) order."""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


UPLEVEL_FREQUENCY = 5
LOW_LEVEL_DECIMATION = 6
SIM_DT = 0.003
BASE_HEIGHT = 0.35
BASE_LINK_NAME = "base_link"
MCL_CAMERA_POS = [0.15, 0.0, 0.20]
MCL_CAMERA_RPY_DEG = (-100.0, 0.0, -90.0)
MCL_CAMERA_ROT_ROS = quat_from_rpy_deg(*MCL_CAMERA_RPY_DEG)
MCL_CAMERA_HEIGHT = 120
MCL_CAMERA_WIDTH = 160
LOUNGE_SCENE_USD = "./exts/scene_data/lounge_usdz/lounge_3dgut_standard_with_ground.usda"
R7_SCENE_USD = "../../3D_files/3DGUT_USDZ/r7_final.usd"
FIRE_EXT_USD = "./exts/scene_data/fire_ext_rigid.usda"
BLUE_BLOCK_USD = "./exts/scene_data/blue_block_rigid.usda"
# r7_final.usd authors GroundPlane at z=0.170092; shift it down so floor collision is z=0.
R7_ASSET_OFFSET = (0.0, 0.0, -0.170092)
R7_TARGET_HEIGHT = 0.0
FIRE_EXT_ORIGINAL_HEIGHT = 0.9965
FIRE_EXT_BOTTOM_TO_ORIGIN = 0.4980
FIRE_EXT_HEIGHT = 0.50
FIRE_EXT_SCALE = FIRE_EXT_HEIGHT / FIRE_EXT_ORIGINAL_HEIGHT
FIRE_EXT_TARGET_OFFSET = (0.0, 0.0, FIRE_EXT_BOTTOM_TO_ORIGIN * FIRE_EXT_SCALE)
BLUE_BLOCK_ORIGINAL_HEIGHT = 0.6431
BLUE_BLOCK_BOTTOM_TO_ORIGIN = 0.3245
BLUE_BLOCK_HEIGHT = 0.20
BLUE_BLOCK_SCALE = BLUE_BLOCK_HEIGHT / BLUE_BLOCK_ORIGINAL_HEIGHT
BLUE_BLOCK_TARGET_OFFSET = (0.0, 0.0, BLUE_BLOCK_BOTTOM_TO_ORIGIN * BLUE_BLOCK_SCALE)
PLAY_TARGET_LIGHT_ENABLED = False
PLAY_FIRE_EXT_LIGHT_INTENSITY_RANGE = (1100.0, 1700.0)
PLAY_BLUE_BLOCK_LIGHT_INTENSITY_RANGE = (900.0, 1500.0)
PLAY_FIRE_EXT_LIGHT_COLOR = (1.0, 0.90, 0.78)
PLAY_BLUE_BLOCK_LIGHT_COLOR = (0.80, 0.88, 1.0)
PLAY_TARGET_LIGHT_HEIGHT = 0.85
PLAY_TARGET_LIGHT_RADIUS = 0.35
PLAY_TARGET_LIGHT_COLOR_JITTER = 0.08
R7_ROBOT_SPAWN_RANGE = {
    "x": (0.4, 0.6),
    "y": (-1.1, -0.9),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (-0.25, 0.25),
}
R7_TARGET_SPAWN_RANGE = {
    "x": (1.2, 3.0),
    "y": (-1.0, 1.0),
    "z": (0.0, 0.0),
    "yaw": (-math.pi, math.pi),
}
INACTIVE_TARGET_POS = (0.0, 0.0, -100.0)
LOUNGE_ASSET_OFFSET = ASSET_OFFSET
# The USDZ wrapper already rotates the imported lounge asset from Y-up to Isaac's Z-up frame.
LOUNGE_ASSET_ROT = (1.0, 0.0, 0.0, 0.0)


@configclass
class MCLSceneCfg(MySceneCfg):
    """MCL scene with an optional robot-mounted camera."""

    front_rgb_camera: TiledCameraCfg | None = None
    physics_ground: AssetBaseCfg | None = None


@configclass
class ActionsCfg:
    joint_pos = mdp.VelocityCommandActionCfg(
        asset_name="robot",
        joint_names=MCLQUAD_SERIAL_JOINT_NAMES,
        preserve_order=True,
        scale={".*HAA": 0.125, "^(?!.*HAA).*": 0.25},
        use_default_offset=True,
        velocity_range=[2.0, 0.5, 0.5],
        policy_dir="../../../../../../low_level_policy/model_4000.pt",
        uplevel_frequency=UPLEVEL_FREQUENCY,
        low_level_decimation=LOW_LEVEL_DECIMATION,
        policy_class_name="ActorCritic",
        policy_obs_dim=45,
        policy_critic_obs_dim=45,
        policy_action_dim=12,
        policy_actor_hidden_dims=[512, 256, 128],
        policy_critic_hidden_dims=[512, 256, 128],
        policy_activation="elu",
        load_actor_only=True,
        velocity_command_start_idx=6,
        use_biarticular_torque_control=True,
        pd_kp=50.0,
        pd_kd=1.5,
        torque_limit=60.0,
        torque_delay_steps=1,
    )

    # // "env": {
    # //     "T": [[1.000, 0.000, 0.000, 0.000], [0.000, 1.000, 0.000, 0.000], [0.000, 0.000, 1.000, 0.000], [0.000, 0.000, 0.000, 1.000]],
    # //     "scale": 1.000000
    # // },

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        gs_image = ObsTerm(
            func=mdp.gs_image_feature,
            params={
                "camera_pos": MCL_CAMERA_POS,
                "camera_rot": [0.0, 0.0, 0.0],
                "asset_offset_pos": LOUNGE_ASSET_OFFSET,
                "asset_offset_rot": LOUNGE_ASSET_ROT,
            },
            noise=None,
        )
        goal_command = ObsTerm(func=mdp.rgb_command, params={"command_name": "rgb_command"}, noise=None)
        actions = ObsTerm(func=mdp.last_action, noise=None)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_bispace,
            params={"joint_names": MCLQUAD_SERIAL_JOINT_NAMES, "preserve_order": True},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel_bispace,
            params={"joint_names": MCLQUAD_SERIAL_JOINT_NAMES, "preserve_order": True},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        root_pos_e = ObsTerm(func=mdp.head_pos_w)
        root_quat_w = ObsTerm(func=mdp.root_quat_w)
        goal_pos = ObsTerm(func=mdp.goal_pos_multi, params={"base_height": BASE_HEIGHT})
        goal_command = ObsTerm(func=mdp.rgb_command, params={"command_name": "rgb_command"}, noise=None)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=None)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=None)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=None)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_bispace,
            params={"joint_names": MCLQUAD_SERIAL_JOINT_NAMES, "preserve_order": True},
            noise=None,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel_bispace,
            params={"joint_names": MCLQUAD_SERIAL_JOINT_NAMES, "preserve_order": True},
            noise=None,
            scale=0.05,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class LocomotionCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.zero_velocity_commands)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_bispace,
            params={"joint_names": MCLQUAD_SERIAL_JOINT_NAMES, "preserve_order": True},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel_bispace,
            params={"joint_names": MCLQUAD_SERIAL_JOINT_NAMES, "preserve_order": True},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.ll_last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: ObsGroup = PolicyCfg()
    critic: ObsGroup = CriticCfg()
    locomotion: ObsGroup = LocomotionCfg()


@configclass
class SeminarCameraObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        camera_image = ObsTerm(
            func=mdp.rgb_image_feature,
            params={
                "sensor_cfg": SceneEntityCfg("front_rgb_camera"),
                "data_type": "rgb",
            },
            noise=None,
        )
        goal_command = ObsTerm(func=mdp.rgb_command, params={"command_name": "rgb_command"}, noise=None)
        actions = ObsTerm(func=mdp.last_action, noise=None)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_bispace,
            params={"joint_names": MCLQUAD_SERIAL_JOINT_NAMES, "preserve_order": True},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel_bispace,
            params={"joint_names": MCLQUAD_SERIAL_JOINT_NAMES, "preserve_order": True},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: ObsGroup = PolicyCfg()
    critic: ObsGroup = ObservationsCfg.CriticCfg()
    locomotion: ObsGroup = ObservationsCfg.LocomotionCfg()


@configclass
class FlatConeCameraObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        camera_image = ObsTerm(
            func=mdp.rgb_image_feature,
            params={
                "sensor_cfg": SceneEntityCfg("front_rgb_camera"),
                "data_type": "rgb",
            },
            noise=None,
        )
        actions = ObsTerm(func=mdp.last_action, noise=None)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_bispace,
            params={"joint_names": MCLQUAD_SERIAL_JOINT_NAMES, "preserve_order": True},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel_bispace,
            params={"joint_names": MCLQUAD_SERIAL_JOINT_NAMES, "preserve_order": True},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        root_pos_e = ObsTerm(func=mdp.head_pos_w)
        root_quat_w = ObsTerm(func=mdp.root_quat_w)
        goal_pos = ObsTerm(func=mdp.goal_pos_single, params={"base_height": BASE_HEIGHT})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=None)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=None)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=None)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_bispace,
            params={"joint_names": MCLQUAD_SERIAL_JOINT_NAMES, "preserve_order": True},
            noise=None,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel_bispace,
            params={"joint_names": MCLQUAD_SERIAL_JOINT_NAMES, "preserve_order": True},
            noise=None,
            scale=0.05,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = PolicyCfg()
    critic: ObsGroup = CriticCfg()
    locomotion: ObsGroup = ObservationsCfg.LocomotionCfg()


@configclass
class RewardsCfg:
    reach_goal = RewTerm(
        func=mdp.reach_goal,
        weight=0.5,
        params={"base_height": BASE_HEIGHT, "command_name": "rgb_command", "threshold": 0.35},
    )
    goal_dis = RewTerm(
        func=mdp.goal_dis, weight=5.0, params={"base_height": BASE_HEIGHT, "command_name": "rgb_command"}
    )
    goal_dis_z = RewTerm(
        func=mdp.goal_dis_z, weight=30.0, params={"base_height": BASE_HEIGHT, "command_name": "rgb_command"}
    )
    goal_heading = RewTerm(
        func=mdp.goal_heading_l1, weight=0.3, params={"base_height": BASE_HEIGHT, "command_name": "rgb_command"}
    )
    stand_still_at_goal = RewTerm(
        func=mdp.stand_still_at_goal, weight=1.0, params={"base_height": BASE_HEIGHT, "command_name": "rgb_command"}
    )
    track_lin_vel_xy_exp_command = RewTerm(
        func=mdp.track_lin_vel_xy_exp_command, weight=0.2, params={"std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp_command = RewTerm(
        func=mdp.track_ang_vel_z_exp_command, weight=0.2, params={"std": math.sqrt(0.25)}
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.002)


@configclass
class FlatConeRewardsCfg:
    reach_goal = RewTerm(
        func=mdp.reach_goal_single,
        weight=1.0,
        params={"base_height": BASE_HEIGHT, "threshold": 0.35},
    )
    goal_dis = RewTerm(func=mdp.goal_dis_single, weight=6.0, params={"base_height": BASE_HEIGHT})
    goal_heading = RewTerm(func=mdp.goal_heading_single_l1, weight=0.9, params={"base_height": BASE_HEIGHT})
    stand_still_at_goal = RewTerm(
        func=mdp.stand_still_at_single_goal,
        weight=1.0,
        params={"base_height": BASE_HEIGHT},
    )
    track_lin_vel_xy_exp_command = RewTerm(
        func=mdp.track_lin_vel_xy_exp_command, weight=0.2, params={"std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp_command = RewTerm(
        func=mdp.track_ang_vel_z_exp_command, weight=0.2, params={"std": math.sqrt(0.25)}
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.002)


@configclass
class CurriculumCfg:
    terrain_levels = None


@configclass
class MCLEventCfg(EventCfg):
    lift_base_after_joint_reset = EventTerm(
        func=mdp.lift_root_state_to_clear_feet,
        mode="reset",
        params={
            "foot_body_names": ".*_foot",
            "min_foot_clearance": 0.01,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class FlatConeEventCfg(MCLEventCfg):
    reset_robot_with_cones = EventTerm(
        func=mdp.reset_single_cone,
        mode="reset",
        params={
            "pose_range": {
                "x": (1.2, 3.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("cone_red"),
        },
    )


@configclass
class MCLQuadGSBaseEnvCfg(UnitreeGo2GSBaseEnvCfg):
    scene: MCLSceneCfg = MCLSceneCfg(num_envs=4096, env_spacing=8)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: MCLEventCfg = MCLEventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    use_biarticular_hip_knee: bool = True
    hip_joint_token: str = "HIP"
    knee_joint_token: str = "KNEE"

    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(1.0 / (UPLEVEL_FREQUENCY * SIM_DT))
        self.episode_length_s = 60.0
        self.sim.dt = SIM_DT
        self.sim.render_interval = LOW_LEVEL_DECIMATION
        self.sim.disable_contact_processing = True
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class MCLQuadGSEnvCfg(MCLQuadGSBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.robot = MCLQUAD_SERIAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/{BASE_LINK_NAME}"
        self.scene.object = AssetBaseCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=LOUNGE_SCENE_USD,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=LOUNGE_ASSET_OFFSET, rot=LOUNGE_ASSET_ROT),
        )
        self.scene.terrain.slope_threshold = 2
        self.scene.terrain.terrain_generator.horizontal_scale = 0.1
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.sub_terrains = {
            "perlin_terrain": terrain_gen.HfPerlinTerrainCfg(horizontal_scale=0.05, frequency=10, zScale=0.0),
        }

        self.actions.joint_pos.scale = {".*HAA": 0.125, "^(?!.*HAA).*": 0.25}
        self.actions.joint_pos.velocity_range = [2.5, 1.0, 1.5]
        self.actions.joint_pos.low_level_decimation = LOW_LEVEL_DECIMATION

        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=BASE_LINK_NAME)
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 1.5)
        self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg("robot", body_names=BASE_LINK_NAME)
        self.events.reset_base.func = mdp.reset_root_state_uniform_with_foot_projection
        self.events.reset_base.params["pose_range"] = {
            "x": (0.4, 0.6),
            "y": (-1.1, -0.9),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.25, 0.25),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["foot_body_names"] = ".*_foot"
        self.events.reset_base.params["min_foot_clearance"] = 0.01
        self.events.reset_base.params["asset_cfg"] = SceneEntityCfg("robot")
        self.events.reset_asset = None
        self.curriculum.terrain_levels = None


@configclass
class MCLQuadSeminarCameraEnvCfg(MCLQuadGSEnvCfg):
    observations: SeminarCameraObservationsCfg = SeminarCameraObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.object.spawn.usd_path = R7_SCENE_USD
        self.scene.object.init_state.pos = R7_ASSET_OFFSET
        # Keep the R7 scene visual, but add an invisible physical floor at the
        # same z=0 height so the robot can stand even if the USD ground is visual-only.
        self.scene.terrain = None
        self.scene.height_scanner = None
        self.scene.physics_ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(
                visible=False,
                size=(20.0, 20.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, R7_TARGET_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        self.scene.cone_red.rigid_objects["cone_red"].spawn.usd_path = FIRE_EXT_USD
        self.scene.cone_red.rigid_objects["cone_red"].spawn.scale = (
            FIRE_EXT_SCALE,
            FIRE_EXT_SCALE,
            FIRE_EXT_SCALE,
        )
        self.scene.cone_red.rigid_objects["cone_red"].init_state.pos = INACTIVE_TARGET_POS
        self.scene.cone_green.rigid_objects["cone_green"].init_state.pos = INACTIVE_TARGET_POS
        self.scene.cone_blue.rigid_objects["cone_blue"].spawn.usd_path = BLUE_BLOCK_USD
        self.scene.cone_blue.rigid_objects["cone_blue"].spawn.scale = (
            BLUE_BLOCK_SCALE,
            BLUE_BLOCK_SCALE,
            BLUE_BLOCK_SCALE,
        )
        self.scene.cone_blue.rigid_objects["cone_blue"].init_state.pos = INACTIVE_TARGET_POS
        self.commands.rgb_command.RGB_prob = [0.5, 0.0, 0.5]
        self.commands.rgb_command.target_asset_names = ["cone_red", None, "cone_blue"]
        self.commands.rgb_command.target_pose_range = R7_TARGET_SPAWN_RANGE.copy()
        self.commands.rgb_command.target_position_offsets = [
            FIRE_EXT_TARGET_OFFSET,
            None,
            BLUE_BLOCK_TARGET_OFFSET,
        ]
        self.commands.rgb_command.inactive_target_pos = INACTIVE_TARGET_POS
        self.events.reset_robot_with_cones = None
        self.events.reset_base.params["pose_range"] = R7_ROBOT_SPAWN_RANGE.copy()
        self.observations.critic.goal_pos.params["base_height"] = R7_TARGET_HEIGHT
        for term_name in ("reach_goal", "goal_dis", "goal_dis_z", "goal_heading", "stand_still_at_goal"):
            getattr(self.rewards, term_name).params["base_height"] = R7_TARGET_HEIGHT
        self.scene.front_rgb_camera = TiledCameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{BASE_LINK_NAME}/front_rgb_camera",
            update_period=self.decimation * self.sim.dt,
            height=MCL_CAMERA_HEIGHT,
            width=MCL_CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=tuple(MCL_CAMERA_POS),
                rot=MCL_CAMERA_ROT_ROS,
                convention="ros",
            ),
        )


@configclass
class MCLQuadFlatConeCameraEnvCfg(MCLQuadGSEnvCfg):
    observations: FlatConeCameraObservationsCfg = FlatConeCameraObservationsCfg()
    rewards: FlatConeRewardsCfg = FlatConeRewardsCfg()
    events: FlatConeEventCfg = FlatConeEventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 64
        self.scene.object = None
        self.scene.cone_green = None
        self.scene.cone_blue = None
        self.scene.front_rgb_camera = TiledCameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{BASE_LINK_NAME}/front_rgb_camera",
            update_period=self.decimation * self.sim.dt,
            height=MCL_CAMERA_HEIGHT,
            width=MCL_CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=tuple(MCL_CAMERA_POS),
                rot=MCL_CAMERA_ROT_ROS,
                convention="ros",
            ),
        )
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat": terrain_gen.HfPerlinTerrainCfg(horizontal_scale=0.05, frequency=10, zScale=0.0),
        }

        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.3, 0.3),
        }
        self.events.reset_asset = None
        self.curriculum.terrain_levels = None


@configclass
class MCLQuadGSEnvCfg_PLAY(MCLQuadGSEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_base.params["pose_range"].update(
            {
                "x": (0.5, 0.5),
                "y": (-1.0, -1.0),
                "yaw": (0.0, 0.0),
            }
        )


@configclass
class MCLQuadSeminarCameraEnvCfg_PLAY(MCLQuadSeminarCameraEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.rgb_command.target_light_enabled = PLAY_TARGET_LIGHT_ENABLED
        self.commands.rgb_command.target_light_intensity_ranges = [
            PLAY_FIRE_EXT_LIGHT_INTENSITY_RANGE,
            None,
            PLAY_BLUE_BLOCK_LIGHT_INTENSITY_RANGE,
        ]
        self.commands.rgb_command.target_light_colors = [
            PLAY_FIRE_EXT_LIGHT_COLOR,
            None,
            PLAY_BLUE_BLOCK_LIGHT_COLOR,
        ]
        self.commands.rgb_command.target_light_height = PLAY_TARGET_LIGHT_HEIGHT
        self.commands.rgb_command.target_light_radius = PLAY_TARGET_LIGHT_RADIUS
        self.commands.rgb_command.target_light_color_jitter = PLAY_TARGET_LIGHT_COLOR_JITTER
        self.events.reset_base.params["pose_range"].update(
            {
                "x": (0.5, 0.5),
                "y": (-1.0, -1.0),
                "yaw": (0.0, 0.0),
            }
        )


@configclass
class MCLQuadFlatConeCameraEnvCfg_PLAY(MCLQuadFlatConeCameraEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_base.params["pose_range"].update(
            {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        )
        self.events.reset_robot_with_cones.params["pose_range"] = {
            "x": (2.0, 2.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }
