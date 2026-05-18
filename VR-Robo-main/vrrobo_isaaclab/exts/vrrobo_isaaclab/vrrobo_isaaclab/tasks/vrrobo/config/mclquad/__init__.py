import gymnasium as gym

from . import agents, mclquad_env_cfg

gym.register(
    id="mclquad_gs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": mclquad_env_cfg.MCLQuadGSEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MCLQuadGSPPORunnerCfg",
    },
)

gym.register(
    id="mclquad_gs_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": mclquad_env_cfg.MCLQuadGSEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MCLQuadGSPPORunnerCfg",
    },
)

gym.register(
    id="mclquad_seminar_camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": mclquad_env_cfg.MCLQuadSeminarCameraEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MCLQuadSeminarCameraPPORunnerCfg",
    },
)

gym.register(
    id="mclquad_seminar_camera_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": mclquad_env_cfg.MCLQuadSeminarCameraEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MCLQuadSeminarCameraPPORunnerCfg",
    },
)

gym.register(
    id="mclquad_flat_cone_camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": mclquad_env_cfg.MCLQuadFlatConeCameraEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MCLQuadFlatConeCameraPPORunnerCfg",
    },
)

gym.register(
    id="mclquad_flat_cone_camera_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": mclquad_env_cfg.MCLQuadFlatConeCameraEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MCLQuadFlatConeCameraPPORunnerCfg",
    },
)
