# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys

from isaaclab.app import AppLauncher

# local imports
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_LOCAL_RL_TRAINING_SRC = os.path.join(_REPO_ROOT, "source", "rl_training")

sys.path.append(os.path.abspath(os.path.join(_THIS_DIR, "..")))
if os.path.isdir(_LOCAL_RL_TRAINING_SRC) and _LOCAL_RL_TRAINING_SRC not in sys.path:
    sys.path.append(_LOCAL_RL_TRAINING_SRC)
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--sim_dt", type=float, default=None, help="Override physics simulation dt in seconds.")
parser.add_argument("--policy_dt", type=float, default=None, help="Override policy step dt in seconds.")
parser.add_argument(
    "--pd-control-mode",
    type=str,
    choices=("effort", "implicit"),
    default=None,
    help="Use custom effort PD or Isaac implicit actuator position drive.",
)
parser.add_argument("--implicit-pd-kp", type=float, default=None, help="Implicit actuator stiffness override.")
parser.add_argument("--implicit-pd-kd", type=float, default=None, help="Implicit actuator damping override.")
parser.add_argument(
    "--implicit-effort-limit",
    type=float,
    default=None,
    help="Implicit actuator effort limit override. Defaults to env_cfg.torque_limit when omitted.",
)
parser.add_argument("--torque-delay-steps", type=int, default=None, help="Override torque delay min/max steps.")
parser.add_argument(
    "--dynamic-conversion",
    type=str,
    choices=("on", "off"),
    default="on",
    help="Enable/disable adding tau_comp (dynamic conversion) to applied torque.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.sim_dt is not None or args_cli.policy_dt is not None:
        sim_dt = float(args_cli.sim_dt) if args_cli.sim_dt is not None else float(env_cfg.sim.dt)
        policy_dt = (
            float(args_cli.policy_dt)
            if args_cli.policy_dt is not None
            else float(env_cfg.decimation) * float(env_cfg.sim.dt)
        )
        decimation_float = policy_dt / sim_dt
        decimation = int(round(decimation_float))
        if decimation <= 0 or abs(decimation_float - decimation) > 1.0e-6:
            raise ValueError(
                f"policy_dt={policy_dt} must be an integer multiple of sim_dt={sim_dt}; "
                f"got decimation={decimation_float}."
            )
        env_cfg.sim.dt = sim_dt
        env_cfg.decimation = decimation
        env_cfg.sim.render_interval = decimation
        scene_cfg = getattr(env_cfg, "scene", None)
        if scene_cfg is not None:
            for sensor_name in ("height_scanner", "height_scanner_base"):
                sensor_cfg = getattr(scene_cfg, sensor_name, None)
                if sensor_cfg is not None:
                    sensor_cfg.update_period = policy_dt
            for sensor_name in (
                "contact_forces",
                "ground_reaction_forces",
                "wall_contact_forces",
                "body_ground_contact_forces",
            ):
                sensor_cfg = getattr(scene_cfg, sensor_name, None)
                if sensor_cfg is not None:
                    sensor_cfg.update_period = sim_dt
        print(
            "[INFO] Timing override applied: sim_dt={:.6f}s, policy_dt={:.6f}s, decimation={}.".format(
                sim_dt,
                policy_dt,
                decimation,
            )
        )
    if args_cli.torque_delay_steps is not None:
        delay_steps = max(0, int(args_cli.torque_delay_steps))
        env_cfg.torque_delay_enable = delay_steps > 0
        env_cfg.torque_delay_steps = delay_steps
        env_cfg.torque_delay_min_steps = delay_steps
        env_cfg.torque_delay_max_steps = delay_steps
        print(f"[INFO] Torque delay override applied: {delay_steps} physics step(s).")
    if args_cli.pd_control_mode is not None:
        env_cfg.pd_control_mode = args_cli.pd_control_mode
        print(f"[INFO] PD control mode override applied: {env_cfg.pd_control_mode}.")
    if getattr(env_cfg, "pd_control_mode", "effort") == "implicit":
        implicit_kp = float(args_cli.implicit_pd_kp) if args_cli.implicit_pd_kp is not None else float(env_cfg.pd_kp)
        implicit_kd = float(args_cli.implicit_pd_kd) if args_cli.implicit_pd_kd is not None else float(env_cfg.pd_kd)
        if args_cli.implicit_effort_limit is not None:
            implicit_effort_limit = float(args_cli.implicit_effort_limit)
        else:
            implicit_effort_limit = getattr(env_cfg, "torque_limit", None)
            implicit_effort_limit = None if implicit_effort_limit is None else float(implicit_effort_limit)
        robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
        if robot_cfg is not None:
            for actuator_cfg in robot_cfg.actuators.values():
                actuator_cfg.stiffness = implicit_kp
                actuator_cfg.damping = implicit_kd
                if implicit_effort_limit is not None:
                    actuator_cfg.effort_limit = implicit_effort_limit
                    actuator_cfg.effort_limit_sim = implicit_effort_limit
        print(
            "[INFO] Implicit actuator PD applied: stiffness={:.6g}, damping={:.6g}, effort_limit={}.".format(
                implicit_kp,
                implicit_kd,
                "default" if implicit_effort_limit is None else f"{implicit_effort_limit:.6g}",
            )
        )
    if hasattr(env_cfg, "dynamic_conversion_enable"):
        env_cfg.dynamic_conversion_enable = args_cli.dynamic_conversion == "on"
        print(
            "[INFO] Training option applied: --dynamic-conversion {} "
            "(tau_comp {}applied).".format(
                args_cli.dynamic_conversion,
                "" if env_cfg.dynamic_conversion_enable else "not ",
            )
        )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed


    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env.reset()

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
