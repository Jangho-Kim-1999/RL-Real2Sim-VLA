# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from rl_training.assets import ISAACLAB_ASSETS_DATA_DIR

# Global motor torque-delay settings consumed by Env_runtime.py.
# Keep the total torque delay at 3 ms with the current 3 ms physics step.
MCLQUAD_SERIAL_TORQUE_DELAY_ENABLE = True
MCLQUAD_SERIAL_TORQUE_DELAY_MIN_STEPS = 1
MCLQUAD_SERIAL_TORQUE_DELAY_MAX_STEPS = 1

# Feedforward terms consumed by Env_runtime.py.
Jm = 1.62e-2
Bm = 9.72e-2

# Default joint posture defined in bi-space for hip-knee pairs.
# - qm = q_hip
# - qb = q_hip + q_knee
DEFAULT_HAA = 0.0
DEFAULT_QM = 0.785
DEFAULT_QB = 2.3558
DEFAULT_KNEE = DEFAULT_QB - DEFAULT_QM

MCLQUAD_SERIAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Quad_v2_serial/Quad_v2_serial _v4.usd",
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Quad_serial/Quad_serial_foot_0.21/Quad_serial_foot_0.21.usd",
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Quad_serial/Quad_serial/Quad_serial.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
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
            armature= Jm,
            viscous_friction= Bm,
        ),
        "HIP": ImplicitActuatorCfg(
            joint_names_expr=[".*HIP"],
            effort_limit=200.0,
            velocity_limit=22,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature= 2 * Jm,
            viscous_friction= 2 * Bm,
        ),
        "KNEE": ImplicitActuatorCfg(
            joint_names_expr=[".*KNEE"],
            effort_limit=200.0,
            velocity_limit=22,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature= Jm,
            viscous_friction= Bm,
        ),
        # "Hip": DelayedPDActuatorCfg(
        #     joint_names_expr=[".*[HAA,HIP]"], 
        #     effort_limit=60.0,
        #     velocity_limit=20.4,
        #     stiffness=0.0,
        #     damping=0.0,
        #     friction=0.0,
        #     armature=0.0,
        #     min_delay=0,
        #     max_delay=0,
        # ),
        # "Knee": DelayedPDActuatorCfg(
        #     joint_names_expr=[".*KNEE"],
        #     effort_limit=60.0,
        #     velocity_limit=20.4,
        #     stiffness=0.0,
        #     damping=0.0,
        #     friction=0.0,
        #     armature=0.0,
        #     min_delay=0,
        #     max_delay=0,
        # ),
    },
)
