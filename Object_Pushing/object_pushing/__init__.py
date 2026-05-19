from __future__ import annotations

import gymnasium as gym

from . import agents, env_cfg


gym.register(
    id="mclquad_object_pushing",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.MCLQuadObjectPushingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MCLQuadObjectPushingPPORunnerCfg",
    },
)

gym.register(
    id="mclquad_object_pushing_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.MCLQuadObjectPushingPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MCLQuadObjectPushingPPORunnerCfg",
    },
)

gym.register(
    id="mclquad_object_pushing_r7_camera_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.MCLQuadObjectPushingR7CameraPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MCLQuadObjectPushingPPORunnerCfg",
    },
)
