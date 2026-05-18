from __future__ import annotations

from isaaclab.utils import configclass
from vrrobo_isaaclab.wrapper.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class MCLQuadObjectPushingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    device = "cuda:0"
    num_steps_per_env = 40
    max_iterations = 10000
    save_interval = 20
    experiment_name = "mclquad_object_pushing"
    empirical_normalization = False
    resume = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.75,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
