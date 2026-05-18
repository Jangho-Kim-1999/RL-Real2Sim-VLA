from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import vrrobo_isaaclab.tasks.vrrobo.mdp as vr_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from vrrobo_isaaclab.assets import MCLQUAD_SERIAL_CFG, MCLQUAD_SERIAL_JOINT_NAMES

from . import mdp


UPLEVEL_FREQUENCY = 5
LOW_LEVEL_DECIMATION = 6
SIM_DT = 0.003
BASE_LINK_NAME = "base_link"
LOW_LEVEL_POLICY_4000 = "../../../../../../low_level_policy/model_4000.pt"


@configclass
class ObjectPushingSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MCLQUAD_SERIAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    push_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PushObject",
        spawn=sim_utils.CylinderCfg(
            radius=mdp.OBJECT_RADIUS,
            height=mdp.OBJECT_HEIGHT,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mdp.OBJECT_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.3,
                dynamic_friction=0.3,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.22, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, mdp.OBJECT_CENTER_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    target_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        spawn=sim_utils.CylinderCfg(
            radius=mdp.OBJECT_RADIUS,
            height=mdp.OBJECT_HEIGHT,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.25), opacity=0.25),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.0, 0.0, mdp.OBJECT_CENTER_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.85, 0.85, 0.85), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.15, 0.15, 0.15), intensity=1000.0),
    )


@configclass
class ActionsCfg:
    joint_pos = vr_mdp.VelocityCommandActionCfg(
        asset_name="robot",
        joint_names=MCLQUAD_SERIAL_JOINT_NAMES,
        preserve_order=True,
        scale={".*HAA": 0.125, "^(?!.*HAA).*": 0.25},
        use_default_offset=True,
        velocity_range=[1.0, 1.0, 1.0],
        policy_dir=LOW_LEVEL_POLICY_4000,
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


@configclass
class ObservationsCfg:
    @configclass
    class PushingPolicyCfg(ObsGroup):
        pushing_state = ObsTerm(
            func=mdp.pushing_policy_observation,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("push_object"),
                "target_cfg": SceneEntityCfg("target_marker"),
                "action_name": "joint_pos",
                "foot_body_names": ".*foot",
            },
            noise=None,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class LocomotionCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=vr_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=vr_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=vr_mdp.zero_velocity_commands)
        joint_pos = ObsTerm(
            func=vr_mdp.joint_pos_rel_bispace,
            params={"joint_names": MCLQUAD_SERIAL_JOINT_NAMES, "preserve_order": True},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=vr_mdp.joint_vel_rel_bispace,
            params={"joint_names": MCLQUAD_SERIAL_JOINT_NAMES, "preserve_order": True},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        actions = ObsTerm(func=vr_mdp.ll_last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: ObsGroup = PushingPolicyCfg()
    critic: ObsGroup = PushingPolicyCfg()
    locomotion: ObsGroup = LocomotionCfg()


@configclass
class CommandsCfg:
    pass


@configclass
class RewardsCfg:
    intrinsic = RewTerm(func=mdp.intrinsic_pushing_reward, weight=1.0)
    front_push = RewTerm(func=mdp.front_push_reward, weight=3.0)
    target_facing = RewTerm(func=mdp.target_facing_reward, weight=1.0)
    object_position = RewTerm(func=mdp.object_position_reward, weight=1.0)
    success_bonus = RewTerm(func=mdp.success_bonus, weight=1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=vr_mdp.time_out, time_out=True)
    object_at_goal = DoneTerm(func=mdp.object_goal_success)
    base_too_low = DoneTerm(func=mdp.base_below_height, params={"minimum_height": 0.15})
    bad_orientation = DoneTerm(func=vr_mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class EventCfg:
    object_geometry = EventTerm(
        func=mdp.randomize_pushing_object_geometry,
        mode="prestartup",
        params={
            "diameter_range": mdp.OBJECT_DIAMETER_RANGE,
            "height_range": mdp.OBJECT_HEIGHT_RANGE,
            "object_cfg": SceneEntityCfg("push_object"),
            "target_cfg": SceneEntityCfg("target_marker"),
        },
    )
    robot_physics_material = EventTerm(
        func=vr_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    object_physics_material = EventTerm(
        func=vr_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("push_object", body_names=".*"),
            "static_friction_range": (0.2, 0.4),
            "dynamic_friction_range": (0.2, 0.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    robot_base_mass = EventTerm(
        func=vr_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    object_mass = EventTerm(
        func=vr_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("push_object", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    reset_robot_joints = EventTerm(
        func=vr_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_scene = EventTerm(func=mdp.reset_pushing_scene, mode="reset")


@configclass
class CurriculumCfg:
    spawn_distances = CurrTerm(
        func=mdp.object_pushing_spawn_distance_curriculum,
        params={
            "steps_per_iteration": 40,
            "iterations_per_level": 100,
            "num_levels": 20,
            "robot_radius_start_range": mdp.ROBOT_RADIUS_START_RANGE,
            "robot_radius_end_range": mdp.ROBOT_RADIUS_END_RANGE,
            "target_distance_start_range": mdp.TARGET_DISTANCE_START_RANGE,
            "target_distance_end_range": mdp.TARGET_DISTANCE_END_RANGE,
            "robot_lateral_start_range": mdp.ROBOT_LATERAL_START_RANGE,
            "robot_lateral_end_range": mdp.ROBOT_LATERAL_END_RANGE,
            "robot_yaw_noise_start_range": mdp.ROBOT_YAW_NOISE_START_RANGE,
            "robot_yaw_noise_end_range": mdp.ROBOT_YAW_NOISE_END_RANGE,
        },
    )


@configclass
class MCLQuadObjectPushingEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectPushingSceneCfg = ObjectPushingSceneCfg(num_envs=300, env_spacing=8.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    use_biarticular_hip_knee: bool = True
    hip_joint_token: str = "HIP"
    knee_joint_token: str = "KNEE"

    def __post_init__(self):
        self.decimation = round(1.0 / (UPLEVEL_FREQUENCY * SIM_DT))
        self.episode_length_s = 30.0
        self.sim.dt = SIM_DT
        self.sim.render_interval = LOW_LEVEL_DECIMATION
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class MCLQuadObjectPushingPlayEnvCfg(MCLQuadObjectPushingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 30.0
        self.observations.locomotion.enable_corruption = False
