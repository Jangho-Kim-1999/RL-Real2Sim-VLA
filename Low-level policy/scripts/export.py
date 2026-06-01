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

"""Export a trained RSL-RL policy as TorchScript for C++ deployment."""

import argparse
import os
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


parser = argparse.ArgumentParser(description="Export an RSL-RL checkpoint to TorchScript.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to instantiate for export.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output .pt path. Defaults to <checkpoint_dir>/exported/policy.pt.",
)

# append RSL-RL and AppLauncher args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Load checkpoint and export policy TorchScript."""
    # Override configurations with CLI arguments.
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1

    # Set seed/device before env creation.
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Resolve checkpoint path.
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Experiment directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    # Resolve export path.
    if args_cli.output is not None:
        output_path = os.path.abspath(args_cli.output)
        export_dir = os.path.dirname(output_path)
        export_file = os.path.basename(output_path)
    else:
        export_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_file = "policy.pt"
        output_path = os.path.join(export_dir, export_file)
    if not export_file.endswith(".pt"):
        raise ValueError(f"--output must end with .pt, got: {export_file}")
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    # Create env and runner to instantiate policy architecture.
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # Keep compatibility with older/newer rsl_rl interfaces.
    try:
        policy_nn = runner.alg.policy  # rsl_rl >= 2.3
    except AttributeError:
        policy_nn = runner.alg.actor_critic  # rsl_rl <= 2.2

    export_policy_as_jit(
        policy=policy_nn,
        normalizer=None,
        path=export_dir,
        filename=export_file,
    )
    print(f"[INFO] Exported TorchScript policy to: {output_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()



# python scripts/export.py \
#   --task Rough-MCLQuad-serial \
#   --checkpoint logs/rsl_rl/MCLrobotics_MCLQuadserial_rough/200hz_260304/model_2999.pt \
#   --headless
