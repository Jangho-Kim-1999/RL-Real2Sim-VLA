"""Run the frozen low-level policy on flat terrain with random velocity commands."""

from __future__ import annotations

import argparse
import math
import os
import sys
import numpy as np

_VRROBO_ISAACLAB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "exts", "vrrobo_isaaclab"))
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "exts", "rsl_rl"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Test the MCL low-level velocity policy without GS/high-level policy.")
parser.add_argument("--task", type=str, default="mclquad_gs_play", help="Task config to use.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps", type=int, default=1200, help="Number of environment steps to run.")
parser.add_argument("--command_interval", type=int, default=100, help="Steps between random command samples.")
parser.add_argument("--print_interval", type=int, default=20, help="Steps between status prints.")
parser.add_argument("--max_vx", type=float, default=0.8, help="Max sampled forward velocity command.")
parser.add_argument("--max_vy", type=float, default=0.4, help="Max sampled lateral velocity command.")
parser.add_argument("--max_wz", type=float, default=0.8, help="Max sampled yaw velocity command.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--no_random", action="store_true", help="Use a fixed command sequence instead of random commands.")
parser.add_argument("--constant_command", action="store_true", help="Use one constant velocity command for the whole run.")
parser.add_argument("--cmd_vx", type=float, default=1.0, help="Constant forward velocity command.")
parser.add_argument("--cmd_vy", type=float, default=0.0, help="Constant lateral velocity command.")
parser.add_argument("--cmd_wz", type=float, default=0.0, help="Constant yaw velocity command.")
parser.add_argument("--with_noise", action="store_true", help="Keep observation/event randomization enabled.")
parser.add_argument(
    "--policy_path",
    type=str,
    default=None,
    help="Optional low-level policy checkpoint path. Relative paths are resolved from vrrobo_isaaclab.",
)
parser.add_argument("--keep_task_terrain", action="store_true", help="Use the task terrain instead of forcing a plane.")
parser.add_argument("--log_dir", type=str, default="logs/ll_diag", help="Directory to save diagnostic .mat file.")
parser.add_argument("--log_steps", type=int, default=500, help="Max RL steps to record detailed physics-level logs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import scipy.io
import torch

import vrrobo_isaaclab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def disable_gs_image_observation(env_cfg) -> None:
    """Remove GS image observation so renderer is not needed for this low-level test."""
    if hasattr(env_cfg.observations, "policy") and hasattr(env_cfg.observations.policy, "gs_image"):
        env_cfg.observations.policy.gs_image = None


def configure_flat_plane(env_cfg) -> None:
    """Match the DEEP flat locomotion setup as closely as this task config allows."""
    terrain = getattr(env_cfg.scene, "terrain", None)
    if terrain is None:
        return
    if hasattr(terrain, "terrain_type"):
        terrain.terrain_type = "plane"
    if hasattr(terrain, "terrain_generator"):
        terrain.terrain_generator = None
    if hasattr(terrain, "max_init_terrain_level"):
        terrain.max_init_terrain_level = None
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None


def configure_deterministic_eval(env_cfg) -> None:
    """Disable noise/randomization that makes low-level policy debugging ambiguous."""
    for group_name in ("policy", "critic", "locomotion"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is not None and hasattr(group, "enable_corruption"):
            group.enable_corruption = False

    if hasattr(env_cfg, "events"):
        for event_name in ("physics_material", "add_base_mass", "base_external_force_torque"):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)
        if hasattr(env_cfg.events, "reset_robot_joints") and env_cfg.events.reset_robot_joints is not None:
            env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
            env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


def configure_policy_path(env_cfg) -> None:
    if args_cli.policy_path is None:
        return
    policy_path = args_cli.policy_path
    if not os.path.isabs(policy_path):
        policy_path = os.path.abspath(os.path.join(_VRROBO_ISAACLAB_DIR, policy_path))
    env_cfg.actions.joint_pos.policy_dir = policy_path


def sample_commands(num_envs: int, device: str, generator: torch.Generator) -> torch.Tensor:
    command = torch.empty(num_envs, 3, device=device)
    command[:, 0].uniform_(-args_cli.max_vx, args_cli.max_vx, generator=generator)
    command[:, 1].uniform_(-args_cli.max_vy, args_cli.max_vy, generator=generator)
    command[:, 2].uniform_(-args_cli.max_wz, args_cli.max_wz, generator=generator)
    return command


def fixed_command(step: int, num_envs: int, device: str) -> torch.Tensor:
    sequence = (
        (0.6, 0.0, 0.0),
        (0.0, 0.3, 0.0),
        (-0.4, 0.0, 0.0),
        (0.0, -0.3, 0.0),
        (0.0, 0.0, 0.6),
        (0.0, 0.0, -0.6),
        (0.0, 0.0, 0.0),
    )
    value = sequence[(step // args_cli.command_interval) % len(sequence)]
    return torch.tensor(value, device=device).repeat(num_envs, 1)


def constant_command(num_envs: int, device: str) -> torch.Tensor:
    value = (args_cli.cmd_vx, args_cli.cmd_vy, args_cli.cmd_wz)
    return torch.tensor(value, device=device).repeat(num_envs, 1)


def command_to_raw_action(command: torch.Tensor, velocity_range: torch.Tensor) -> torch.Tensor:
    normalized = torch.clamp(command / velocity_range, -0.95, 0.95)
    return 0.5 * torch.log((1.0 + normalized) / (1.0 - normalized))


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    disable_gs_image_observation(env_cfg)
    if not args_cli.keep_task_terrain:
        configure_flat_plane(env_cfg)
    configure_policy_path(env_cfg)
    if not args_cli.with_noise:
        configure_deterministic_eval(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    obs, _ = env.reset()

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    action_term = base_env.action_manager.get_term("joint_pos")
    velocity_range = action_term.velocity_range
    foot_body_ids, _ = robot.find_bodies(".*_foot", preserve_order=True)
    generator = torch.Generator(device=base_env.device)
    generator.manual_seed(args_cli.seed)

    # --- Enable diagnostics on action term ---
    action_term._diag_enabled = True
    action_term._diag = {
        "ll_obs": [], "ll_output": [], "ll_phys_step": [],
        "q": [], "dq": [], "q_des": [],
        "tau_computed": [], "tau_applied": [], "phys_step": [],
    }
    action_term._diag_phys_step = 0

    # --- RL-step level buffers (all envs) ---
    rl_vel_cmd = []
    rl_vel_measured = []
    rl_base_z = []
    rl_min_foot_z = []
    rl_base_quat = []
    rl_ang_vel_b = []
    rl_joint_pos = []
    rl_joint_vel = []
    rl_raw_actions = []       # LL last raw output (env 0)
    rl_reset_mask = []        # bool: any reset this step?

    command = torch.zeros(base_env.num_envs, 3, device=base_env.device)
    raw_action = command_to_raw_action(command, velocity_range)

    print("[INFO] Low-level policy random velocity test")
    print(f"[INFO] task={args_cli.task}, envs={base_env.num_envs}, steps={args_cli.steps}")
    print(f"[INFO] policy={env_cfg.actions.joint_pos.policy_dir}")
    print(f"[INFO] terrain={getattr(env_cfg.scene.terrain, 'terrain_type', None)}")
    print(f"[INFO] velocity_range={velocity_range.detach().cpu().tolist()}")
    print(f"[INFO] joint_names={list(action_term._joint_names)}")
    print(f"[INFO] joint_ids={action_term._resolve_joint_id_list()}")
    print(f"[INFO] Detailed physics-level log: first {args_cli.log_steps} RL steps → {args_cli.log_dir}")
    print("[INFO] columns: step | cmd(vx,vy,wz) | measured(vx,vy,wz) | abs_error | base_z | min_foot_z")

    with torch.inference_mode():
        for step in range(args_cli.steps):
            # Stop physics-level logging after log_steps to save memory
            if step == args_cli.log_steps:
                action_term._diag_enabled = False

            if step % args_cli.command_interval == 0:
                if args_cli.constant_command:
                    command = constant_command(base_env.num_envs, base_env.device)
                elif args_cli.no_random:
                    command = fixed_command(step, base_env.num_envs, base_env.device)
                else:
                    command = sample_commands(base_env.num_envs, base_env.device, generator)
                raw_action = command_to_raw_action(command, velocity_range)

            obs, rew, terminated, truncated, extras = env.step(raw_action)

            # --- Collect RL-step data ---
            measured = torch.cat(
                [robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], dim=1
            )
            base_z = robot.data.root_pos_w[:, 2]
            min_foot_z = robot.data.body_link_pos_w[:, foot_body_ids, 2].amin(dim=1)
            any_reset = bool((terminated | truncated).any())

            rl_vel_cmd.append(action_term.velocity_command.detach().cpu().numpy().copy())
            rl_vel_measured.append(measured.detach().cpu().numpy().copy())
            rl_base_z.append(base_z.detach().cpu().numpy().copy())
            rl_min_foot_z.append(min_foot_z.detach().cpu().numpy().copy())
            rl_base_quat.append(robot.data.root_quat_w.detach().cpu().numpy().copy())
            rl_ang_vel_b.append(robot.data.root_ang_vel_b.detach().cpu().numpy().copy())
            rl_joint_pos.append(robot.data.joint_pos.detach().cpu().numpy().copy())
            rl_joint_vel.append(robot.data.joint_vel.detach().cpu().numpy().copy())
            rl_raw_actions.append(action_term.raw_actions[0].detach().cpu().numpy().copy())
            rl_reset_mask.append(any_reset)

            if step % args_cli.print_interval == 0:
                error = torch.abs(action_term.velocity_command - measured)
                print(
                    f"{step:05d} | "
                    f"{action_term.velocity_command[0].detach().cpu().numpy()} | "
                    f"{measured[0].detach().cpu().numpy()} | "
                    f"{error[0].detach().cpu().numpy()} | "
                    f"{base_z[0].item():.3f} | "
                    f"{min_foot_z[0].item():.3f}"
                )

            if any_reset:
                reset_ids = torch.where(terminated | truncated)[0].detach().cpu().tolist()
                print(f"[INFO] reset at step={step}, env_ids={reset_ids}")

    env.close()

    # --- Save diagnostic data ---
    _save_diagnostics(
        log_dir=args_cli.log_dir,
        action_term=action_term,
        rl_data={
            "vel_cmd": np.array(rl_vel_cmd),
            "vel_measured": np.array(rl_vel_measured),
            "base_z": np.array(rl_base_z),
            "min_foot_z": np.array(rl_min_foot_z),
            "base_quat": np.array(rl_base_quat),
            "ang_vel_b": np.array(rl_ang_vel_b),
            "joint_pos": np.array(rl_joint_pos),
            "joint_vel": np.array(rl_joint_vel),
            "raw_actions_env0": np.array(rl_raw_actions),
            "reset_mask": np.array(rl_reset_mask, dtype=np.uint8),
        },
        env_cfg=env_cfg,
        sim_dt=env_cfg.sim.dt,
        decimation=base_env.cfg.decimation,
        ll_decimation=action_term.low_level_decimation,
    )


def _save_diagnostics(log_dir, action_term, rl_data, env_cfg, sim_dt, decimation, ll_decimation):
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, "ll_diagnostics.mat")

    diag = action_term._diag

    def _to_np(lst):
        return np.array(lst) if lst else np.array([])

    phys_dt = sim_dt
    rl_dt = sim_dt * decimation

    mat_data = {
        # --- Metadata ---
        "sim_dt": np.float64(sim_dt),
        "rl_dt": np.float64(rl_dt),
        "ll_decimation": np.float64(ll_decimation),
        "velocity_range": action_term.velocity_range.cpu().numpy(),
        "joint_ids": np.asarray(action_term._resolve_joint_id_list(), dtype=np.int32),
        "joint_names": np.asarray(list(action_term._joint_names), dtype=object),

        # --- RL-step level (N_rl, num_envs, dim) ---
        # time axis: t_rl[i] = i * rl_dt
        "vel_cmd":      rl_data["vel_cmd"],       # (N_rl, num_envs, 3)
        "vel_measured": rl_data["vel_measured"],  # (N_rl, num_envs, 3)
        "base_z":       rl_data["base_z"],        # (N_rl, num_envs)
        "min_foot_z":   rl_data["min_foot_z"],    # (N_rl, num_envs)
        "base_quat":    rl_data["base_quat"],     # (N_rl, num_envs, 4)
        "ang_vel_b":    rl_data["ang_vel_b"],     # (N_rl, num_envs, 3)
        "joint_pos_rl": rl_data["joint_pos"],     # (N_rl, num_envs, 12)
        "joint_vel_rl": rl_data["joint_vel"],     # (N_rl, num_envs, 12)
        "raw_actions_env0": rl_data["raw_actions_env0"],  # (N_rl, 12) LL last output env 0
        "reset_mask":   rl_data["reset_mask"],    # (N_rl,)

        # --- Physics-step level (env 0 only) ---
        # t_phys[i] = phys_step[i] * sim_dt
        "phys_step":    _to_np(diag["phys_step"]),     # (N_phys,)
        "q_phys":       _to_np(diag["q"]),             # (N_phys, 12) actual joint pos
        "dq_phys":      _to_np(diag["dq"]),            # (N_phys, 12) actual joint vel
        "q_des_phys":   _to_np(diag["q_des"]),         # (N_phys, 12) desired joint pos
        "tau_computed": _to_np(diag["tau_computed"]),  # (N_phys, 12) PD output before delay
        "tau_applied":  _to_np(diag["tau_applied"]),   # (N_phys, 12) torque after delay queue

        # --- LL-step level (env 0 only, every low_level_decimation phys steps) ---
        # t_ll[i] = ll_phys_step[i] * sim_dt
        "ll_phys_step": _to_np(diag["ll_phys_step"]),  # (N_ll,)
        "ll_obs":       _to_np(diag["ll_obs"]),         # (N_ll, 45)
        "ll_output":    _to_np(diag["ll_output"]),      # (N_ll, 12)
    }

    scipy.io.savemat(save_path, mat_data)
    print(f"[INFO] Diagnostics saved → {save_path}")
    print(f"[INFO]   RL steps logged:    {rl_data['vel_cmd'].shape[0]}")
    print(f"[INFO]   Phys steps logged:  {len(diag['phys_step'])}")
    print(f"[INFO]   LL calls logged:    {len(diag['ll_phys_step'])}")
    print(f"[INFO]   ll_obs shape:       {_to_np(diag['ll_obs']).shape}")
    print(f"[INFO]   tau_applied shape:  {_to_np(diag['tau_applied']).shape}")


if __name__ == "__main__":
    main()
    simulation_app.close()
