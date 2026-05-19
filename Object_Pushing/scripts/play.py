from __future__ import annotations

import argparse
import csv
import os
import sys

_OBJECT_PUSHING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WORKSPACE_DIR = os.path.abspath(os.path.join(_OBJECT_PUSHING_DIR, ".."))
_VRROBO_ISAACLAB_DIR = os.path.join(_WORKSPACE_DIR, "VR-Robo-main", "vrrobo_isaaclab")
sys.path.insert(0, _OBJECT_PUSHING_DIR)
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "exts", "vrrobo_isaaclab"))
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "exts", "rsl_rl"))
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "scripts", "rsl_rl"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play the MCLQuad object pushing task.")
parser.add_argument("--video", action="store_true", default=False, help="Record a play video.")
parser.add_argument("--video_length", type=int, default=300, help="Recorded video length in high-level steps.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="mclquad_object_pushing_play", help="Gym task id.")
parser.add_argument("--no_checkpoint", action="store_true", help="Run with zero high-level actions.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many high-level steps. 0 means no limit.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--log_dir", type=str, default=None, help="Directory for CSV/MATLAB play logs.")
parser.add_argument("--log_file", type=str, default="object_pushing_play_log.csv", help="CSV log filename.")
parser.add_argument("--log_every", type=int, default=1, help="Log one row every N high-level steps.")
parser.add_argument("--log_limit", type=int, default=0, help="Maximum number of logged rows. 0 means unlimited.")
parser.add_argument("--log_env_id", type=int, default=0, help="Environment index to log.")
parser.add_argument("--log_mat", action="store_true", help="Also save a MATLAB .mat file. Requires scipy.")
parser.add_argument("--log_mat_file", type=str, default="object_pushing_play_log.mat", help="MATLAB .mat filename.")
parser.add_argument(
    "--plot_matlab_script_file",
    type=str,
    default="plot_object_pushing_play_log.m",
    help="MATLAB plotting script filename.",
)
parser.add_argument(
    "--object_xy_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Override object spawn local X/Y range in meters. Example: --object_xy_range -0.5 0.5",
)
parser.add_argument(
    "--object_yaw_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Override initial object yaw range in radians.",
)
parser.add_argument(
    "--robot_radius_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Override initial robot-object distance range in meters.",
)
parser.add_argument(
    "--target_distance_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Override initial object-target distance range in meters.",
)
parser.add_argument(
    "--target_angle_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Override target direction angle range around the object in radians.",
)
parser.add_argument(
    "--robot_lateral_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Override robot lateral offset from object-target line in meters.",
)
parser.add_argument(
    "--robot_yaw_noise_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Override robot yaw noise around target direction in radians.",
)
parser.add_argument(
    "--disable_spawn_curriculum",
    action="store_true",
    help="Disable spawn-distance curriculum during play and use reset ranges directly.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = args_cli.video

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import object_pushing  # noqa: F401
from object_pushing.plain_rsl_rl_wrapper import PlainRslRlVecEnvWrapper


LOG_FIELDS = [
    "step",
    "time_s",
    "episode_length",
    "reward",
    "done",
    "success",
    "object_target_dist_xy",
    "object_x",
    "object_y",
    "object_z",
    "target_x",
    "target_y",
    "target_z",
    "robot_x",
    "robot_y",
    "robot_z",
    "object_vx",
    "object_vy",
    "object_vz",
    "action_raw_vx",
    "action_raw_vy",
    "action_raw_wz",
    "velocity_cmd_vx",
    "velocity_cmd_vy",
    "velocity_cmd_wz",
]


def _as_env_np(value: torch.Tensor, env_id: int) -> np.ndarray:
    return value.detach()[env_id].cpu().numpy().copy()


def _capture_metrics(env: PlainRslRlVecEnvWrapper, step: int, actions: torch.Tensor, env_id: int) -> dict[str, float]:
    scene = env.unwrapped.scene
    robot = scene["robot"]
    push_object = scene["push_object"]
    target = scene["target_marker"]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")

    object_pos = _as_env_np(push_object.data.root_pos_w, env_id)
    target_pos = _as_env_np(target.data.root_pos_w, env_id)
    robot_pos = _as_env_np(robot.data.root_pos_w, env_id)
    if hasattr(push_object.data, "root_lin_vel_w"):
        object_vel = _as_env_np(push_object.data.root_lin_vel_w, env_id)
    else:
        object_vel = _as_env_np(push_object.data.root_vel_w[:, :3], env_id)
    raw_action = _as_env_np(actions, env_id)
    velocity_cmd = torch.tanh(actions.detach()[env_id]) * action_term.velocity_range
    velocity_cmd = velocity_cmd.cpu().numpy().copy()
    dist_xy = float(np.linalg.norm(object_pos[:2] - target_pos[:2]))
    sim_dt = float(getattr(env.unwrapped.cfg.sim, "dt", 0.0))
    decimation = int(getattr(env.unwrapped.cfg, "decimation", 1))

    return {
        "step": int(step),
        "time_s": float(step * sim_dt * decimation),
        "episode_length": int(env.unwrapped.episode_length_buf[env_id].item()),
        "reward": 0.0,
        "done": 0,
        "success": int(dist_xy < 0.05),
        "object_target_dist_xy": dist_xy,
        "object_x": float(object_pos[0]),
        "object_y": float(object_pos[1]),
        "object_z": float(object_pos[2]),
        "target_x": float(target_pos[0]),
        "target_y": float(target_pos[1]),
        "target_z": float(target_pos[2]),
        "robot_x": float(robot_pos[0]),
        "robot_y": float(robot_pos[1]),
        "robot_z": float(robot_pos[2]),
        "object_vx": float(object_vel[0]),
        "object_vy": float(object_vel[1]),
        "object_vz": float(object_vel[2]),
        "action_raw_vx": float(raw_action[0]),
        "action_raw_vy": float(raw_action[1]),
        "action_raw_wz": float(raw_action[2]),
        "velocity_cmd_vx": float(velocity_cmd[0]),
        "velocity_cmd_vy": float(velocity_cmd[1]),
        "velocity_cmd_wz": float(velocity_cmd[2]),
    }


def _write_matlab_plot_script(script_path: str, csv_path: str) -> None:
    csv_name = os.path.basename(csv_path)
    script = f"""clear; clc; close all;
log_file = fullfile(fileparts(mfilename('fullpath')), '{csv_name}');
T = readtable(log_file);

figure('Name', 'Object Pushing Play Log', 'Color', 'w');
tiledlayout(2, 2);

nexttile;
plot(T.time_s, T.object_target_dist_xy, 'LineWidth', 1.5); hold on;
yline(0.05, '--r', 'success threshold');
grid on;
xlabel('time [s]');
ylabel('object-target XY dist [m]');
title('Position Error');

nexttile;
plot(T.object_x, T.object_y, 'b', 'LineWidth', 1.5); hold on;
plot(T.target_x, T.target_y, 'g--', 'LineWidth', 1.2);
plot(T.robot_x, T.robot_y, 'k:', 'LineWidth', 1.0);
axis equal; grid on;
xlabel('x [m]');
ylabel('y [m]');
legend('object', 'target', 'robot', 'Location', 'best');
title('XY Trajectory');

nexttile;
plot(T.time_s, T.reward, 'LineWidth', 1.2); grid on;
xlabel('time [s]');
ylabel('reward');
title('Reward');

nexttile;
plot(T.time_s, [T.velocity_cmd_vx, T.velocity_cmd_vy, T.velocity_cmd_wz], 'LineWidth', 1.1);
grid on;
xlabel('time [s]');
ylabel('command');
legend('v_x', 'v_y', 'w_z', 'Location', 'best');
title('High-Level Velocity Command');
"""
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)


def _save_mat_file(mat_path: str, rows: list[dict[str, float]]) -> bool:
    if not rows:
        return False
    try:
        from scipy.io import savemat
    except ImportError:
        print("[WARN] scipy is not available; skipped .mat export. CSV and MATLAB plot script were still written.")
        return False
    mat_data = {field: np.asarray([row[field] for row in rows]) for field in LOG_FIELDS}
    savemat(mat_path, mat_data, do_compression=True)
    return True


def _range_arg(value: list[float] | None) -> tuple[float, float] | None:
    if value is None:
        return None
    range_min = float(min(value))
    range_max = float(max(value))
    return range_min, range_max


def _apply_spawn_overrides(env_cfg) -> None:
    reset_params = env_cfg.events.reset_scene.params
    if reset_params is None:
        reset_params = {}
        env_cfg.events.reset_scene.params = reset_params

    curriculum_params = None
    spawn_curriculum = getattr(env_cfg.curriculum, "spawn_distances", None)
    if spawn_curriculum is not None:
        curriculum_params = spawn_curriculum.params
        if curriculum_params is None:
            curriculum_params = {}
            spawn_curriculum.params = curriculum_params

    overrides = {
        "object_xy_range": _range_arg(args_cli.object_xy_range),
        "object_yaw_range": _range_arg(args_cli.object_yaw_range),
        "robot_radius_range": _range_arg(args_cli.robot_radius_range),
        "target_distance_range": _range_arg(args_cli.target_distance_range),
        "target_angle_range": _range_arg(args_cli.target_angle_range),
        "robot_lateral_range": _range_arg(args_cli.robot_lateral_range),
        "robot_yaw_noise_range": _range_arg(args_cli.robot_yaw_noise_range),
    }
    active_overrides = {key: value for key, value in overrides.items() if value is not None}
    reset_params.update(active_overrides)

    curriculum_range_map = {
        "robot_radius_range": ("robot_radius_start_range", "robot_radius_end_range"),
        "target_distance_range": ("target_distance_start_range", "target_distance_end_range"),
        "robot_lateral_range": ("robot_lateral_start_range", "robot_lateral_end_range"),
        "robot_yaw_noise_range": ("robot_yaw_noise_start_range", "robot_yaw_noise_end_range"),
    }
    if args_cli.disable_spawn_curriculum:
        env_cfg.curriculum.spawn_distances = None
    elif curriculum_params is not None:
        for reset_key, curriculum_keys in curriculum_range_map.items():
            if overrides[reset_key] is None:
                continue
            for curriculum_key in curriculum_keys:
                curriculum_params[curriculum_key] = overrides[reset_key]

    if active_overrides:
        print("[INFO] Play spawn range overrides:")
        for key, value in active_overrides.items():
            print(f"       {key}={value}")
        if args_cli.disable_spawn_curriculum:
            print("[INFO] Spawn curriculum disabled for play.")
        else:
            print("[INFO] Matching curriculum start/end ranges were overridden for fixed play ranges.")
    elif args_cli.disable_spawn_curriculum:
        env_cfg.curriculum.spawn_distances = None
        print("[INFO] Spawn curriculum disabled for play.")


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _apply_spawn_overrides(env_cfg)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.join(_OBJECT_PUSHING_DIR, "logs", "rsl_rl", agent_cfg.experiment_name)
    if args_cli.no_checkpoint:
        resume_path = None
        log_dir = os.path.join(log_root_path, "debug_no_checkpoint")
    else:
        if not os.path.isdir(log_root_path):
            raise FileNotFoundError(
                f"No trained checkpoint directory found: {log_root_path}\n"
                "Use --no_checkpoint for a smoke test, or run train.py first."
            )
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording play video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = PlainRslRlVecEnvWrapper(env)

    if args_cli.no_checkpoint:
        print("[INFO] Running without checkpoint; high-level actions are zero.")

        def policy(_obs):
            return torch.zeros(env.num_envs, env.num_actions, device=env.device)

    else:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    play_log_dir = args_cli.log_dir or os.path.join(log_dir, "play_logs")
    os.makedirs(play_log_dir, exist_ok=True)
    csv_path = os.path.join(play_log_dir, args_cli.log_file)
    mat_path = os.path.join(play_log_dir, args_cli.log_mat_file)
    matlab_script_path = os.path.join(play_log_dir, args_cli.plot_matlab_script_file)
    log_every = max(1, int(args_cli.log_every))
    log_limit = max(0, int(args_cli.log_limit))
    log_env_id = int(np.clip(args_cli.log_env_id, 0, env.num_envs - 1))
    logged_rows: list[dict[str, float]] = []
    print(f"[INFO] Logging env {log_env_id} play metrics to: {csv_path}")

    obs, _ = env.get_observations()
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            should_log = step % log_every == 0 and (log_limit == 0 or len(logged_rows) < log_limit)
            row = _capture_metrics(env, step, actions, log_env_id) if should_log else None
            obs, rewards, dones, _ = env.step(actions)
            if row is not None:
                row["reward"] = float(rewards[log_env_id].item())
                row["done"] = int(dones[log_env_id].item())
                logged_rows.append(row)
        step += 1
        if step % 25 == 0:
            print(
                f"[INFO] step={step} reward0={float(rewards[0]):.3f} "
                f"done0={bool(dones[0])}"
            )
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            break

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(logged_rows)
    _write_matlab_plot_script(matlab_script_path, csv_path)
    if args_cli.log_mat:
        if _save_mat_file(mat_path, logged_rows):
            print(f"[INFO] Saved MATLAB MAT log: {mat_path}")
    print(f"[INFO] Saved CSV log: {csv_path}")
    print(f"[INFO] Saved MATLAB plot script: {matlab_script_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
