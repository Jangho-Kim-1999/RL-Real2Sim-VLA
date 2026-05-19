from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

_OBJECT_PUSHING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WORKSPACE_DIR = os.path.abspath(os.path.join(_OBJECT_PUSHING_DIR, ".."))
_VRROBO_ISAACLAB_DIR = os.path.join(_WORKSPACE_DIR, "VR-Robo-main", "vrrobo_isaaclab")
sys.path.insert(0, _OBJECT_PUSHING_DIR)
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "exts", "vrrobo_isaaclab"))
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "exts", "rsl_rl"))
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "scripts", "rsl_rl"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(
    description="Collect NaVILA-style VLA fine-tuning data from the trained MCLQuad object-pushing expert."
)
parser.add_argument("--task", type=str, default="mclquad_object_pushing_r7_camera_play", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Collector currently supports one env.")
parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to attempt.")
parser.add_argument("--start_episode_id", type=int, default=0, help="Episode id offset used in saved directory names.")
parser.add_argument("--max_steps", type=int, default=150, help="High-level steps per episode.")
parser.add_argument("--save_dir", type=str, default=os.path.join(_OBJECT_PUSHING_DIR, "vla_push_dataset"))
parser.add_argument(
    "--success_distance",
    type=float,
    default=0.05,
    help="Object-target XY success threshold in meters.",
)
parser.add_argument("--camera_name", type=str, default="front_rgb_camera", help="Scene camera sensor name.")
parser.add_argument("--num_history", type=int, default=7, help="History frames before the current frame.")
parser.add_argument("--max_merge", type=int, default=3, help="Maximum consecutive same-direction actions to merge.")
parser.add_argument("--seed", type=int, default=7, help="Environment seed and instruction RNG seed.")
parser.add_argument(
    "--no_checkpoint",
    action="store_true",
    help="Run zero high-level actions for a pipeline smoke test.",
)
parser.add_argument("--overwrite", action="store_true", help="Delete save_dir before collecting.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--object_xy_range", type=float, nargs=2, default=(-0.5, 0.5), metavar=("MIN", "MAX"))
parser.add_argument("--robot_radius_range", type=float, nargs=2, default=(0.6, 1.0), metavar=("MIN", "MAX"))
parser.add_argument("--target_distance_range", type=float, nargs=2, default=(1.0, 3.0), metavar=("MIN", "MAX"))
parser.add_argument(
    "--target_angle_range",
    type=float,
    nargs=2,
    default=(-3.14159265, 3.14159265),
    metavar=("MIN", "MAX"),
)
parser.add_argument("--robot_lateral_range", type=float, nargs=2, default=(-0.05, 0.05), metavar=("MIN", "MAX"))
parser.add_argument("--robot_yaw_noise_range", type=float, nargs=2, default=(-0.1, 0.1), metavar=("MIN", "MAX"))
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    raise ValueError("collect_vla_push_data.py currently supports --num_envs 1 only.")
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import object_pushing  # noqa: F401
from object_pushing.plain_rsl_rl_wrapper import PlainRslRlVecEnvWrapper
from object_pushing.vla_dataset import (
    action_records_to_dict,
    compute_action_record,
    generate_instruction,
    merge_consecutive_actions,
    merged_actions_to_dict,
    save_episode_dataset,
    summarize_samples,
    to_pil_rgb,
    write_json,
    yaw_from_quat_wxyz,
)


def _sorted_range(values: tuple[float, float] | list[float]) -> tuple[float, float]:
    return float(min(values)), float(max(values))


def _load_json_if_exists(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_collector_env_overrides(env_cfg) -> None:
    """Use broad data-collection spawn ranges and prevent success auto-reset."""
    env_cfg.curriculum.spawn_distances = None
    if hasattr(env_cfg.terminations, "object_at_goal"):
        env_cfg.terminations.object_at_goal = None
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    reset_params = env_cfg.events.reset_scene.params
    if reset_params is None:
        reset_params = {}
        env_cfg.events.reset_scene.params = reset_params
    reset_params.update(
        {
            "object_xy_range": _sorted_range(args_cli.object_xy_range),
            "robot_radius_range": _sorted_range(args_cli.robot_radius_range),
            "target_distance_range": _sorted_range(args_cli.target_distance_range),
            "target_angle_range": _sorted_range(args_cli.target_angle_range),
            "robot_lateral_range": _sorted_range(args_cli.robot_lateral_range),
            "robot_yaw_noise_range": _sorted_range(args_cli.robot_yaw_noise_range),
        }
    )


def _torch_to_numpy(value: torch.Tensor, env_id: int = 0) -> np.ndarray:
    return value.detach()[env_id].cpu().numpy().copy()


def _robot_pose_xy_yaw(base_env, env_id: int = 0) -> tuple[float, float, float]:
    robot = base_env.scene["robot"]
    pos = _torch_to_numpy(robot.data.root_pos_w, env_id)
    quat = _torch_to_numpy(robot.data.root_quat_w, env_id)
    return float(pos[0]), float(pos[1]), yaw_from_quat_wxyz(quat)


def _object_target_distance(base_env, env_id: int = 0) -> float:
    push_object = base_env.scene["push_object"]
    target = base_env.scene["target_marker"]
    object_pos = _torch_to_numpy(push_object.data.root_pos_w, env_id)
    target_pos = _torch_to_numpy(target.data.root_pos_w, env_id)
    return float(np.linalg.norm(object_pos[:2] - target_pos[:2]))


def _capture_camera_rgb(base_env, camera_name: str, env_id: int = 0):
    sensor = base_env.scene.sensors.get(camera_name)
    if sensor is None:
        raise RuntimeError(
            f"Camera sensor {camera_name!r} was not found. Use the mclquad_object_pushing_r7_camera_play task."
        )
    if "rgb" not in sensor.data.output:
        base_env.sim.render()
    if "rgb" not in sensor.data.output:
        raise RuntimeError(f"Camera sensor {camera_name!r} has no rgb output.")
    image = sensor.data.output["rgb"][env_id]
    if torch.is_tensor(image):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
            image = image.permute(1, 2, 0)
        image = image.numpy()
    return to_pil_rgb(image)


def _make_policy(env: PlainRslRlVecEnvWrapper, agent_cfg, resume_path: str | None):
    if args_cli.no_checkpoint:
        print("[INFO] Running without checkpoint; high-level actions are zero.")

        def policy(_obs):
            return torch.zeros(env.num_envs, env.num_actions, device=env.device)

        return policy

    print(f"[INFO] Loading object-pushing expert checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    return runner.get_inference_policy(device=env.unwrapped.device)


def _build_summary(save_dir: Path, attempted: int, success_count: int, episode_summaries, all_samples):
    return {
        "save_dir": str(save_dir),
        "annotations_file": "annotations.json",
        "image_folder_for_navila": str(save_dir),
        "attempted_this_run": int(attempted),
        "success_this_run": int(success_count),
        "total_successful_episodes": len(episode_summaries),
        "episodes": episode_summaries,
        **summarize_samples(all_samples),
    }


def _collect_episode(env: PlainRslRlVecEnvWrapper, policy, episode_id: int, rng: random.Random):
    obs, _ = env.reset()
    base_env = env.unwrapped
    base_env.sim.render()

    instruction = generate_instruction(rng)
    frames = []
    action_records = []
    success = False
    final_distance = float("inf")

    for step in range(max(1, int(args_cli.max_steps))):
        pose_before = _robot_pose_xy_yaw(base_env)
        frames.append(_capture_camera_rgb(base_env, args_cli.camera_name))

        with torch.inference_mode():
            actions = policy(obs)
            obs, _rewards, dones, _extras = env.step(actions)

        pose_after = _robot_pose_xy_yaw(base_env)
        record = compute_action_record(step, pose_before, pose_after)
        action_records.append(record)

        final_distance = _object_target_distance(base_env)
        if final_distance < float(args_cli.success_distance):
            frames.append(_capture_camera_rgb(base_env, args_cli.camera_name))
            action_records.append(
                compute_action_record(
                    step + 1,
                    _robot_pose_xy_yaw(base_env),
                    _robot_pose_xy_yaw(base_env),
                )
            )
            success = True
            break

        if bool(dones[0].item()):
            break

    if not success:
        return None

    merged_actions = merge_consecutive_actions(action_records, max_merge=args_cli.max_merge)
    metadata = {
        "success": True,
        "final_object_target_distance_m": final_distance,
        "raw_actions": action_records_to_dict(action_records),
        "merged_actions": merged_actions_to_dict(merged_actions),
    }
    return {
        "episode_id": int(episode_id),
        "instruction": instruction,
        "frames": frames,
        "action_records": action_records,
        "merged_actions": merged_actions,
        "metadata": metadata,
    }


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    save_dir = Path(args_cli.save_dir)
    if args_cli.overwrite and save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = save_dir / "annotations.json"
    summary_path = save_dir / "dataset_summary.json"
    all_annotations = _load_json_if_exists(annotations_path, [])
    all_samples = _load_json_if_exists(save_dir / "samples_index.json", [])
    episode_summaries = _load_json_if_exists(summary_path, {}).get("episodes", [])

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _apply_collector_env_overrides(env_cfg)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join(_OBJECT_PUSHING_DIR, "logs", "rsl_rl", agent_cfg.experiment_name)
    resume_path = None
    if not args_cli.no_checkpoint:
        if not os.path.isdir(log_root_path):
            raise FileNotFoundError(
                f"No trained checkpoint directory found: {log_root_path}\n"
                "Use --no_checkpoint for a smoke test, or run Object_Pushing/scripts/train.py first."
            )
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = PlainRslRlVecEnvWrapper(env)
    policy = _make_policy(env, agent_cfg, resume_path)

    rng = random.Random(args_cli.seed)
    attempted = 0
    success_count = 0
    start_id = int(args_cli.start_episode_id)
    try:
        for local_ep in range(max(0, int(args_cli.episodes))):
            if not simulation_app.is_running():
                break
            episode_id = start_id + local_ep
            attempted += 1
            result = _collect_episode(env, policy, episode_id, rng)
            if result is None:
                print(f"[INFO] episode={episode_id:04d} failed")
                continue

            samples, annotations = save_episode_dataset(
                save_dir,
                result["episode_id"],
                result["frames"],
                result["merged_actions"],
                result["instruction"],
                num_history=args_cli.num_history,
                metadata=result["metadata"],
            )
            all_samples.extend(samples)
            all_annotations.extend(annotations)
            success_count += 1
            episode_summaries.append(
                {
                    "episode_id": result["episode_id"],
                    "success": True,
                    "num_frames": len(result["frames"]),
                    "num_samples": len(samples),
                    "instruction": result["instruction"],
                    "final_object_target_distance_m": result["metadata"]["final_object_target_distance_m"],
                }
            )

            write_json(annotations_path, all_annotations)
            write_json(save_dir / "samples_index.json", all_samples)
            write_json(
                summary_path,
                _build_summary(save_dir, attempted, success_count, episode_summaries, all_samples),
            )
            print(
                f"[INFO] episode={episode_id:04d} success samples={len(samples)} "
                f"total_success={len(episode_summaries)} total_samples={len(all_samples)}"
            )
    finally:
        env.close()

    write_json(annotations_path, all_annotations)
    write_json(save_dir / "samples_index.json", all_samples)
    write_json(summary_path, _build_summary(save_dir, attempted, success_count, episode_summaries, all_samples))
    print(f"[INFO] Saved NaVILA annotations: {annotations_path}")
    print(f"[INFO] Saved dataset summary: {summary_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
