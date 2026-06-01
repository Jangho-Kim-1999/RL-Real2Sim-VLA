# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fine-tuning entrypoint with CLI options (e.g., --noise off)."""

import argparse
import math
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_LOCAL_RL_TRAINING_SRC = os.path.join(_REPO_ROOT, "source", "rl_training")

sys.path.append(os.path.abspath(os.path.join(_THIS_DIR, "..")))
if os.path.isdir(_LOCAL_RL_TRAINING_SRC) and _LOCAL_RL_TRAINING_SRC not in sys.path:
    sys.path.append(_LOCAL_RL_TRAINING_SRC)
import cli_args

# ---------------------------------------------------------------------------
# Fine-tuning options (edit here instead of adding CLI flags)
# ---------------------------------------------------------------------------
# Set each entry to `None` to keep the task default.
FINETUNING_COMMAND_RANGES = {
    "lin_vel_x": (-2.5, 2.5),
    "lin_vel_y": (-0.8, 0.8),
    "ang_vel_z": (-0.8, 0.8),
}


def _normalize_range(name: str, value):
    """Validate and normalize optional range overrides."""
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"FINETUNING_COMMAND_RANGES['{name}'] must be None or (min, max), got: {value}")
    low = float(value[0])
    high = float(value[1])
    if low > high:
        raise ValueError(f"Invalid range for '{name}': min={low} is greater than max={high}.")
    return (low, high)


# add argparse arguments
parser = argparse.ArgumentParser(description="Fine-tune RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument(
    "--total_timesteps",
    type=int,
    default=None,
    help="Target total environment timesteps. Overrides --max_iterations when set.",
)
parser.add_argument(
    "--noise",
    type=str,
    choices=("on", "off"),
    default="on",
    help="Observation corruption/noise for policy observations.",
)
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
cli_args.configure_process_determinism(args_cli)

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
import torch

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
    """Fine-tune with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # Fine-tuning options from CLI.
    if args_cli.noise == "off":
        if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.enable_corruption = False
        print("[INFO] Fine-tuning option applied: --noise off (policy observation corruption disabled).")
    else:
        if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.enable_corruption = True
        print("[INFO] Fine-tuning option applied: --noise on (policy observation corruption enabled).")

    if hasattr(env_cfg, "dynamic_conversion_enable"):
        env_cfg.dynamic_conversion_enable = args_cli.dynamic_conversion == "on"
        print(
            "[INFO] Fine-tuning option applied: --dynamic-conversion {} "
            "(tau_comp {}applied).".format(
                args_cli.dynamic_conversion,
                "" if env_cfg.dynamic_conversion_enable else "not ",
            )
        )

    # Fine-tuning command range options from this file.
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        ranges = env_cfg.commands.base_velocity.ranges
        command_overrides = {
            "lin_vel_x": _normalize_range("lin_vel_x", FINETUNING_COMMAND_RANGES.get("lin_vel_x")),
            "lin_vel_y": _normalize_range("lin_vel_y", FINETUNING_COMMAND_RANGES.get("lin_vel_y")),
            "ang_vel_z": _normalize_range("ang_vel_z", FINETUNING_COMMAND_RANGES.get("ang_vel_z")),
        }
        if command_overrides["lin_vel_x"] is not None:
            ranges.lin_vel_x = command_overrides["lin_vel_x"]
        if command_overrides["lin_vel_y"] is not None:
            ranges.lin_vel_y = command_overrides["lin_vel_y"]
        if command_overrides["ang_vel_z"] is not None:
            ranges.ang_vel_z = command_overrides["ang_vel_z"]
        print(
            "[INFO] Fine-tuning command ranges:"
            f" lin_vel_x={ranges.lin_vel_x},"
            f" lin_vel_y={ranges.lin_vel_y},"
            f" ang_vel_z={ranges.ang_vel_z}"
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


    # Optional total timestep target -> convert to max_iterations.
    if args_cli.total_timesteps is not None:
        if args_cli.total_timesteps <= 0:
            raise ValueError(f"--total_timesteps must be > 0, got {args_cli.total_timesteps}")
        steps_per_iter = int(env_cfg.scene.num_envs) * int(agent_cfg.num_steps_per_env)
        if steps_per_iter <= 0:
            raise ValueError(f"Invalid steps_per_iter={steps_per_iter}. Check num_envs/num_steps_per_env.")
        computed_iterations = max(1, math.ceil(args_cli.total_timesteps / steps_per_iter))
        print(
            "[INFO] Converted --total_timesteps={} to max_iterations={} "
            "(num_envs={} * num_steps_per_env={} = {} steps/iter).".format(
                args_cli.total_timesteps,
                computed_iterations,
                env_cfg.scene.num_envs,
                agent_cfg.num_steps_per_env,
                steps_per_iter,
            )
        )
        agent_cfg.max_iterations = computed_iterations

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

# Example:
# python scripts/finetunning.py \
#   --task Flat-MCLQuad-serial \
#   --headless \
#   --resume \
#   --load_run 2026-03-24_10-23-36_2order_50hz \
#   --checkpoint model_2000.pt \
#   --max_iterations 2000
