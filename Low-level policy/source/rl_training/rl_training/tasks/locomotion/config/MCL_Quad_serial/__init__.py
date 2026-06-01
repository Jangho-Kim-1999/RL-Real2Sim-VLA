# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

##
# Register Gym environments.
##
gym.register(
    id="Flat-MCLQuad-serial",
    entry_point="rl_training.tasks.locomotion.Env_runtime:RuntimeTorqueManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MCLQuadserialFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:MCLQuadserialFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{__name__}.cusrl_ppo_cfg:MCLQuadserialFlatTrainerCfg",
    },
)

gym.register(
    id="Rough-MCLQuad-serial",
    entry_point="rl_training.tasks.locomotion.Env_runtime:RuntimeTorqueManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:MCLQuadserialRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:MCLQuadserialRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{__name__}.cusrl_ppo_cfg:MCLQuadserialRoughTrainerCfg",
    },
)

gym.register(
    id="Block-MCLQuad-serial",
    entry_point="rl_training.tasks.locomotion.Env_runtime:RuntimeTorqueManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.block_env_cfg:MCLQuadserialBlockEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:MCLQuadserialBlockPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{__name__}.cusrl_ppo_cfg:MCLQuadserialFlatTrainerCfg",
    },
)

gym.register(
    id="FRPush-MCLQuad-serial",
    entry_point="rl_training.tasks.locomotion.Env_runtime:RuntimeTorqueManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr_push_env_cfg:MCLQuadserialFRPushEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:MCLQuadserialFRPushPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{__name__}.cusrl_ppo_cfg:MCLQuadserialFlatTrainerCfg",
    },
)

gym.register(
    id="AgileJump-MCLQuad-serial",
    entry_point="rl_training.tasks.locomotion.Env_runtime:RuntimeTorqueManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.agile_jump_env:MCLQuadserialAgileJumpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:MCLQuadserialAgileJumpPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{__name__}.cusrl_ppo_cfg:MCLQuadserialFlatTrainerCfg",
    },
)

gym.register(
    id="ForwardJump-MCLQuad-serial",
    entry_point="rl_training.tasks.locomotion.Env_runtime:RuntimeTorqueManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forward_jump_env_cfg:MCLQuadserialForwardJumpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:MCLQuadserialAgileJumpPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{__name__}.cusrl_ppo_cfg:MCLQuadserialFlatTrainerCfg",
    },
)

gym.register(
    id="Stand-MCLQuad-serial",
    entry_point="rl_training.tasks.locomotion.Env_runtime:RuntimeTorqueManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_env_cfg:MCLQuadserialStandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:MCLQuadserialStandPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{__name__}.cusrl_ppo_cfg:MCLQuadserialFlatTrainerCfg",
    },
)

gym.register(
    id="BodyPose-MCLQuad-serial",
    entry_point="rl_training.tasks.locomotion.Env_runtime:RuntimeTorqueManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.body_pose_env_cfg:MCLQuadserialBodyPoseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:MCLQuadserialBodyPosePPORunnerCfg",
        "cusrl_cfg_entry_point": f"{__name__}.cusrl_ppo_cfg:MCLQuadserialFlatTrainerCfg",
    },
)

gym.register(
    id="Attitude-MCLQuad-serial",
    entry_point="rl_training.tasks.locomotion.Env_runtime:RuntimeTorqueManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attitude_env_cfg:MCLQuadserialAttitudeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:MCLQuadserialAttitudePPORunnerCfg",
        "cusrl_cfg_entry_point": f"{__name__}.cusrl_ppo_cfg:MCLQuadserialFlatTrainerCfg",
    },
)
