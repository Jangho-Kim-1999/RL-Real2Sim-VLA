from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from vrrobo_isaaclab.assets import ISAACLAB_ASSETS_DATA_DIR

Jm = 1.62e-2
Bm = 9.72e-2

DEFAULT_HAA = 0.0
DEFAULT_QM = 0.785
DEFAULT_QB = 2.3558
DEFAULT_KNEE = DEFAULT_QB - DEFAULT_QM

MCLQUAD_SERIAL_JOINT_NAMES = [
    "FLHAA",
    "FLHIP",
    "FLKNEE",
    "FRHAA",
    "FRHIP",
    "FRKNEE",
    "RLHAA",
    "RLHIP",
    "RLKNEE",
    "RRHAA",
    "RRHIP",
    "RRKNEE",
]

MCLQUAD_SERIAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Quad_serial/Quad_serial_foot_0.21/Quad_serial_foot_0.21.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*HAA": DEFAULT_HAA,
            ".*HIP": DEFAULT_QM,
            ".*KNEE": DEFAULT_KNEE,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    actuators={
        "HAA": ImplicitActuatorCfg(
            joint_names_expr=[".*HAA"],
            effort_limit=200.0,
            velocity_limit=22,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=Jm,
            viscous_friction=Bm,
        ),
        "HIP": ImplicitActuatorCfg(
            joint_names_expr=[".*HIP"],
            effort_limit=200.0,
            velocity_limit=22,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=Jm,
            viscous_friction=Bm,
        ),
        "KNEE": ImplicitActuatorCfg(
            joint_names_expr=[".*KNEE"],
            effort_limit=200.0,
            velocity_limit=22,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=Jm,
            viscous_friction=Bm,
        ),
    },
)
