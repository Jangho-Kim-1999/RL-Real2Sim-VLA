# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class MCLQuadserialRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "MCLrobotics_MCLQuadserial_rough"
    logger = "tensorboard"
    empirical_normalization = False
    clip_actions = 100
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MCLQuadserialFlatPPORunnerCfg(MCLQuadserialRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 3000
        self.experiment_name = "MCLrobotics_MCLQuadserial_flat"


@configclass
class MCLQuadserialBlockPPORunnerCfg(MCLQuadserialFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.experiment_name = "MCLrobotics_MCLQuadserial_block"


@configclass
class MCLQuadserialFRPushPPORunnerCfg(MCLQuadserialFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = 48
        self.max_iterations = 4000
        self.experiment_name = "MCLrobotics_MCLQuadserial_fr_push"


@configclass
class MCLQuadserialAgileJumpPPORunnerCfg(MCLQuadserialFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = 96
        self.max_iterations = 4000
        self.experiment_name = "MCLrobotics_MCLQuadserial_agile_jump"


@configclass
class MCLQuadserialStandPPORunnerCfg(MCLQuadserialFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.experiment_name = "MCLrobotics_MCLQuadserial_stand"


@configclass
class MCLQuadserialBodyPosePPORunnerCfg(MCLQuadserialFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.experiment_name = "MCLrobotics_MCLQuadserial_body_pose"


@configclass
class MCLQuadserialAttitudePPORunnerCfg(MCLQuadserialFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.experiment_name = "MCLrobotics_MCLQuadserial_attitude"
