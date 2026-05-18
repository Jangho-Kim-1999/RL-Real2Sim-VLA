"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import os
import sys

_VRROBO_ISAACLAB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "exts", "vrrobo_isaaclab"))
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "exts", "rsl_rl"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--no_checkpoint", action="store_true", help="Run with zero actions without loading a policy.")
parser.add_argument("--keyboard", action="store_true", help="Use keyboard velocity commands as high-level actions.")
parser.add_argument("--vla", action="store_true", help="Use NaVILA VLA as the high-level policy.")
parser.add_argument(
    "--vla_repo",
    type=str,
    default=os.environ.get(
        "NAVILA_REPO", os.path.abspath(os.path.join(_VRROBO_ISAACLAB_DIR, "..", "..", "NaVILA-main"))
    ),
    help="Path to the local NaVILA repository.",
)
parser.add_argument(
    "--vla_model_path",
    type=str,
    default=os.environ.get("NAVILA_MODEL_PATH", "a8cheng/navila-llama3-8b-8f"),
    help="NaVILA checkpoint path or Hugging Face model id.",
)
parser.add_argument("--vla_model_base", type=str, default=None, help="Optional base model path for LoRA checkpoints.")
parser.add_argument("--vla_instruction", type=str, default=None, help="Fixed navigation instruction for NaVILA.")
parser.add_argument(
    "--vla_instruction_template",
    type=str,
    default="Navigate to the {target}.",
    help="Instruction template used with the env rgb_command target when --vla_instruction is not set.",
)
parser.add_argument(
    "--vla_target_names",
    type=str,
    default="red target,green target,blue target",
    help="Comma-separated target names for rgb_command indices.",
)
parser.add_argument("--vla_camera_name", type=str, default="front_rgb_camera", help="Camera sensor name for NaVILA.")
parser.add_argument("--vla_num_video_frames", type=int, default=8, help="Number of image frames passed to NaVILA.")
parser.add_argument("--vla_update_every", type=int, default=1, help="Minimum high-level steps between NaVILA calls.")
parser.add_argument("--vla_forward_speed", type=float, default=0.5, help="Velocity for parsed move-forward primitives.")
parser.add_argument("--vla_turn_speed", type=float, default=0.8, help="Yaw rate for parsed turn primitives.")
parser.add_argument(
    "--vla_default_forward_distance",
    type=float,
    default=0.25,
    help="Fallback distance when NaVILA says to move forward without a number.",
)
parser.add_argument(
    "--vla_default_turn_degrees",
    type=float,
    default=30.0,
    help="Fallback turn angle when NaVILA says to turn without a number.",
)
parser.add_argument("--vla_max_hold_steps", type=int, default=60, help="Maximum high-level steps for one parsed primitive.")
parser.add_argument("--vla_load_4bit", action="store_true", help="Load NaVILA with bitsandbytes 4-bit quantization.")
parser.add_argument("--vla_load_8bit", action="store_true", help="Load NaVILA with bitsandbytes 8-bit quantization.")
parser.add_argument("--vla_device", type=str, default="cuda", help="Device passed to NaVILA model loading.")
parser.add_argument("--vla_device_map", type=str, default="auto", help="Device map passed to NaVILA model loading.")
parser.add_argument("--vla_max_new_tokens", type=int, default=96, help="Maximum generated tokens per NaVILA call.")
parser.add_argument("--vla_log_file", type=str, default=None, help="Optional CSV path for NaVILA text/action logs.")
parser.add_argument("--save_camera_png", action="store_true", help="Save front_rgb_camera frames as PNG files.")
parser.add_argument("--camera_png_dir", type=str, default=None, help="Directory for saved camera PNG files.")
parser.add_argument("--camera_png_every", type=int, default=1, help="Save one PNG every N simulation steps.")
parser.add_argument(
    "--camera_png_interval_s",
    type=float,
    default=0.0,
    help="Save one PNG every N seconds. Overrides --camera_png_every when positive.",
)
parser.add_argument("--camera_png_limit", type=int, default=200, help="Maximum number of PNG files to save.")
parser.add_argument("--log_mat", action="store_true", help="Save play diagnostics to a MATLAB .mat file.")
parser.add_argument("--log_mat_dir", type=str, default=None, help="Directory for MATLAB play diagnostics.")
parser.add_argument("--log_mat_file", type=str, default="play_log.mat", help="MATLAB diagnostics filename.")
parser.add_argument(
    "--plot_matlab_script_file",
    type=str,
    default="plot_play_log.m",
    help="MATLAB plotting script written next to the .mat diagnostics file.",
)
parser.add_argument("--log_mat_every", type=int, default=1, help="Log one high-level step every N steps.")
parser.add_argument("--log_mat_limit", type=int, default=0, help="Maximum high-level log samples. 0 means unlimited.")
parser.add_argument("--log_csv", action="store_true", help="Deprecated; CSV diagnostics are always saved.")
parser.add_argument("--log_csv_dir", type=str, default=None, help="Directory for CSV play diagnostics.")
parser.add_argument("--log_csv_file", type=str, default="play_log.csv", help="CSV diagnostics filename.")
parser.add_argument("--log_csv_every", type=int, default=1, help="Log one high-level CSV row every N steps.")
parser.add_argument("--log_csv_limit", type=int, default=0, help="Maximum high-level CSV samples. 0 means unlimited.")
parser.add_argument("--log_env_id", type=int, default=0, help="Environment index to log.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop play after this many high-level env steps. 0 means no limit.")
parser.add_argument("--max_time_s", type=float, default=0.0, help="Stop play after this many simulated seconds. 0 means no limit.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import carb.input
import gymnasium as gym
import numpy as np
import omni.appwindow
from PIL import Image

# Import extensions to set up environment tasks
import vrrobo_isaaclab.tasks  # noqa: F401
from carb.input import KeyboardEventType
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from vrrobo_isaaclab.wrapper import RslRlGSEnvWrapper, RslRlOnPolicyRunnerCfg

from rsl_rl.runners import OnPolicyRunner

MOVE_CAMERA = False

try:
    from isaaclab_tasks.utils.wrappers.rsl_rl import export_policy_as_jit, export_policy_as_onnx
except ModuleNotFoundError:
    export_policy_as_jit = None
    export_policy_as_onnx = None


def main():
    """Play with RSL-RL agent."""
    if args_cli.vla and args_cli.keyboard:
        raise ValueError("--vla and --keyboard both replace the high-level policy; use only one at a time.")
    if args_cli.vla_load_4bit and args_cli.vla_load_8bit:
        raise ValueError("Use only one of --vla_load_4bit or --vla_load_8bit.")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.vla:
        resume_path = None
        log_dir = os.path.join(log_root_path, "navila_vla")
    elif args_cli.no_checkpoint:
        resume_path = None
        log_dir = os.path.join(log_root_path, "debug_no_checkpoint")
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlGSEnvWrapper(env)

    vla_policy = None
    if args_cli.vla:
        from vrrobo_isaaclab.vla import NaVILAVelocityPolicy

        os.makedirs(log_dir, exist_ok=True)
        target_names = tuple(name.strip() for name in args_cli.vla_target_names.split(",") if name.strip())
        vla_log_file = args_cli.vla_log_file or os.path.join(log_dir, "navila_vla_outputs.csv")
        print("[INFO]: Running with NaVILA VLA high-level policy.")
        print(f"[INFO]: NaVILA repo: {args_cli.vla_repo}")
        print(f"[INFO]: NaVILA model: {args_cli.vla_model_path}")
        print(f"[INFO]: NaVILA log: {vla_log_file}")
        vla_policy = NaVILAVelocityPolicy(
            env,
            navila_repo=args_cli.vla_repo,
            model_path=args_cli.vla_model_path,
            model_base=args_cli.vla_model_base,
            instruction=args_cli.vla_instruction,
            instruction_template=args_cli.vla_instruction_template,
            target_names=target_names,
            camera_name=args_cli.vla_camera_name,
            num_video_frames=args_cli.vla_num_video_frames,
            update_every=args_cli.vla_update_every,
            forward_speed=args_cli.vla_forward_speed,
            turn_speed=args_cli.vla_turn_speed,
            default_forward_distance=args_cli.vla_default_forward_distance,
            default_turn_degrees=args_cli.vla_default_turn_degrees,
            max_hold_steps=args_cli.vla_max_hold_steps,
            device=args_cli.vla_device,
            device_map=args_cli.vla_device_map,
            load_4bit=args_cli.vla_load_4bit,
            load_8bit=args_cli.vla_load_8bit,
            max_new_tokens=args_cli.vla_max_new_tokens,
            log_path=vla_log_file,
        )
        policy = vla_policy

    elif args_cli.no_checkpoint:
        print("[INFO]: Running without checkpoint; applying zero high-level actions unless --keyboard is set.")

        def policy(_obs):
            return torch.zeros(env.num_envs, env.num_actions, device=env.device)

    else:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        # if "unitree_go2_gs" not in log_root_path:
        ppo_runner.load(resume_path)

        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # export policy to onnx/jit when the Isaac Lab helper exists in this installation
        if export_policy_as_jit is not None and export_policy_as_onnx is not None:
            export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
            export_policy_as_jit(
                ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
            )
            export_policy_as_onnx(
                ppo_runner.alg.actor_critic,
                normalizer=ppo_runner.obs_normalizer,
                path=export_model_dir,
                filename="policy.onnx",
            )
    # camera_direction = [2, 2, -4]
    camera_direction = np.array([5.0, 0.0, -6.0])

    keyboard_input = carb.input.KeyboardInput
    pressed_movement_keys = {
        keyboard_input.W: False,
        keyboard_input.S: False,
        keyboard_input.A: False,
        keyboard_input.D: False,
        keyboard_input.F: False,
        keyboard_input.G: False,
    }

    def refresh_keyboard_command():
        env.override_command[:, 0] = (
            0.8 * float(pressed_movement_keys[keyboard_input.W])
            - 0.8 * float(pressed_movement_keys[keyboard_input.S])
        )
        env.override_command[:, 1] = (
            0.5 * float(pressed_movement_keys[keyboard_input.A])
            - 0.5 * float(pressed_movement_keys[keyboard_input.D])
        )
        env.override_command[:, 2] = (
            1.0 * float(pressed_movement_keys[keyboard_input.F])
            - 1.0 * float(pressed_movement_keys[keyboard_input.G])
        )

    def on_keyboard_input(e):
        if e.input in pressed_movement_keys:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                pressed_movement_keys[e.input] = True
                refresh_keyboard_command()
            elif e.type == KeyboardEventType.KEY_RELEASE:
                pressed_movement_keys[e.input] = False
                refresh_keyboard_command()
        if e.input == keyboard_input.X:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                for key in pressed_movement_keys:
                    pressed_movement_keys[key] = False
                env.override_command[:] = 0
        if e.input == keyboard_input.N:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                if env.unwrapped.scene.terrain is not None:
                    env.unwrapped.scene.terrain.terrain_types[:] -= 1
                    env.unwrapped.scene.terrain.terrain_types[:] = torch.clip(
                        env.unwrapped.scene.terrain.terrain_types[:], 0, 19
                    )
        if e.input == keyboard_input.M:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                if env.unwrapped.scene.terrain is not None:
                    env.unwrapped.scene.terrain.terrain_types[:] += 1
                    env.unwrapped.scene.terrain.terrain_types[:] = torch.clip(
                        env.unwrapped.scene.terrain.terrain_types[:], 0, 19
                    )

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input = carb.input.acquire_input_interface()
    keyboard_sub = input.subscribe_to_keyboard_events(keyboard, on_keyboard_input)
    if args_cli.keyboard:
        print("[INFO]: Keyboard control enabled: W/S forward/back, A/D left/right, F/G yaw, X stop.")

    sim_dt = float(getattr(env.unwrapped.cfg.sim, "dt", np.nan))
    decimation = int(getattr(env.unwrapped.cfg, "decimation", 1))
    play_step_dt = float(getattr(env.unwrapped, "step_dt", sim_dt * decimation))
    max_play_steps = max(0, int(args_cli.max_steps))
    if args_cli.max_time_s > 0.0:
        max_steps_from_time = max(1, int(np.ceil(float(args_cli.max_time_s) / max(play_step_dt, 1e-9))))
        max_play_steps = min(max_play_steps, max_steps_from_time) if max_play_steps > 0 else max_steps_from_time
        print(f"[INFO]: Play will stop after {max_play_steps} steps (~{max_play_steps * play_step_dt:.2f}s).")
    elif max_play_steps > 0:
        print(f"[INFO]: Play will stop after {max_play_steps} steps (~{max_play_steps * play_step_dt:.2f}s).")

    camera_png_dir = args_cli.camera_png_dir or os.path.join(log_dir, "camera_png")
    if args_cli.camera_png_interval_s > 0.0:
        camera_png_every = max(1, int(round(float(args_cli.camera_png_interval_s) / max(play_step_dt, 1e-9))))
    else:
        camera_png_every = max(1, int(args_cli.camera_png_every))
    camera_png_limit = int(args_cli.camera_png_limit)
    camera_png_count = 0
    camera_png_warned = False
    if args_cli.save_camera_png:
        os.makedirs(camera_png_dir, exist_ok=True)
        print(
            f"[INFO]: Saving camera PNG frames to: {camera_png_dir} "
            f"(every {camera_png_every} steps, ~{camera_png_every * play_step_dt:.2f}s)"
        )

    mat_log_dir = args_cli.log_mat_dir or os.path.join(log_dir, "matlab_logs")
    mat_log_path = os.path.join(mat_log_dir, args_cli.log_mat_file)
    mat_log_every = max(1, int(args_cli.log_mat_every))
    mat_log_limit = int(args_cli.log_mat_limit)

    csv_log_dir = args_cli.log_csv_dir or os.path.join(log_dir, "csv_logs")
    if args_cli.log_csv_file == "play_log.csv":
        if resume_path is not None:
            checkpoint_stem = os.path.splitext(os.path.basename(resume_path))[0]
            csv_file = f"play_{checkpoint_stem}.csv"
        elif args_cli.vla:
            csv_file = "play_navila_vla.csv"
        else:
            csv_file = "play_debug_no_checkpoint.csv"
    else:
        csv_file = args_cli.log_csv_file if args_cli.log_csv_file.endswith(".csv") else f"{args_cli.log_csv_file}.csv"
    csv_log_path = os.path.join(csv_log_dir, csv_file)
    csv_log_every = max(1, int(args_cli.log_csv_every))
    csv_log_limit = int(args_cli.log_csv_limit)
    csv_logging_enabled = True

    diag_log_enabled = args_cli.log_mat or csv_logging_enabled
    active_log_every = [mat_log_every] if args_cli.log_mat else []
    active_log_every += [csv_log_every] if csv_logging_enabled else []
    diag_log_every = min(active_log_every) if active_log_every else 1
    active_limits = [limit for limit in (mat_log_limit if args_cli.log_mat else 0, csv_log_limit if csv_logging_enabled else 0) if limit > 0]
    diag_log_limit = max(active_limits) if active_limits else 0

    mat_log_env_id = int(np.clip(args_cli.log_env_id, 0, env.num_envs - 1))
    mat_log_count = 0
    mat_log_saved = False
    action_term = None
    robot_asset = None
    mat_log: dict[str, list[np.ndarray | float | int | bool]] = {
        "policy_step": [],
        "t_policy_s": [],
        "episode_length": [],
        "policy_action_raw": [],
        "velocity_command_b": [],
        "reward": [],
        "done": [],
        "root_pos_w": [],
        "root_quat_w": [],
        "root_lin_vel_w": [],
        "root_ang_vel_w": [],
        "root_lin_vel_b": [],
        "root_ang_vel_b": [],
        "projected_gravity_b": [],
        "joint_pos": [],
        "joint_vel": [],
        "joint_pos_target": [],
        "root_yaw_w": [],
        "root_heading_w": [],
        "cone_red_pos_w": [],
        "cone_red_pos_env": [],
        "cone_green_pos_w": [],
        "cone_green_pos_env": [],
        "cone_blue_pos_w": [],
        "cone_blue_pos_env": [],
        "active_target_idx": [],
        "active_target_pos_w": [],
        "active_target_pos_env": [],
    }

    def _tensor_env0(value, env_id: int = mat_log_env_id):
        if value is None:
            return np.array([])
        if torch.is_tensor(value):
            value = value.detach()
            if value.ndim > 0 and value.shape[0] > env_id:
                value = value[env_id]
            return value.cpu().numpy().copy()
        value = np.asarray(value)
        if value.ndim > 0 and value.shape[0] > env_id:
            value = value[env_id]
        return value.copy()

    def _scalar_env0(value, env_id: int = mat_log_env_id):
        sampled = _tensor_env0(value, env_id)
        if np.asarray(sampled).size == 0:
            return np.nan
        return float(np.asarray(sampled).reshape(-1)[0])

    def _get_scene_asset(name: str):
        try:
            return env.unwrapped.scene[name]
        except Exception:
            return None

    def _get_object_pos_w(asset):
        if asset is None:
            return np.full(3, np.nan, dtype=np.float32)
        data = getattr(asset, "data", None)
        if data is None:
            return np.full(3, np.nan, dtype=np.float32)
        if hasattr(data, "object_pos_w"):
            pos = data.object_pos_w
        elif hasattr(data, "root_pos_w"):
            pos = data.root_pos_w
        else:
            return np.full(3, np.nan, dtype=np.float32)
        pos = _tensor_env0(pos)
        return np.asarray(pos).reshape(-1, 3)[0]

    def _yaw_from_quat_wxyz(quat) -> float:
        quat = np.asarray(quat, dtype=np.float64).reshape(-1)
        if quat.size < 4 or not np.all(np.isfinite(quat[:4])):
            return np.nan
        w, x, y, z = quat[:4]
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _get_active_target_idx() -> int:
        try:
            command = env.unwrapped.command_manager.get_command("rgb_command")
            command_env = _tensor_env0(command)
            if command_env.size == 0:
                return -1
            return int(np.argmax(command_env.reshape(-1)[:3]))
        except Exception:
            return -1

    if diag_log_enabled:
        if args_cli.log_mat:
            os.makedirs(mat_log_dir, exist_ok=True)
        os.makedirs(csv_log_dir, exist_ok=True)
        try:
            action_term = env.unwrapped.action_manager.get_term("joint_pos")
        except Exception as exc:
            print(f"[WARN]: Failed to find action term 'joint_pos' for play diagnostics: {exc}")
            action_term = None
        robot_asset = _get_scene_asset("robot")
        if args_cli.log_mat and action_term is not None and hasattr(action_term, "_diag_enabled"):
            action_term._diag_enabled = True
            action_term._diag_phys_step = 0
            for key in action_term._diag:
                action_term._diag[key].clear()
        if args_cli.log_mat:
            print(f"[INFO]: MATLAB diagnostics enabled: {mat_log_path} (env_id={mat_log_env_id})")
        print(f"[INFO]: CSV diagnostics enabled: {csv_log_path} (env_id={mat_log_env_id})")

    def append_mat_log(step: int, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor):
        nonlocal mat_log_count
        if not diag_log_enabled:
            return
        if step % diag_log_every != 0:
            return
        if diag_log_limit > 0 and mat_log_count >= diag_log_limit:
            if action_term is not None and hasattr(action_term, "_diag_enabled"):
                action_term._diag_enabled = False
            save_mat_log()
            save_csv_log()
            return

        step_dt = play_step_dt
        cone_red_pos_w = _get_object_pos_w(_get_scene_asset("cone_red"))
        cone_green_pos_w = _get_object_pos_w(_get_scene_asset("cone_green"))
        cone_blue_pos_w = _get_object_pos_w(_get_scene_asset("cone_blue"))
        target_positions_w = [cone_red_pos_w, cone_green_pos_w, cone_blue_pos_w]
        active_target_idx = _get_active_target_idx()
        if 0 <= active_target_idx < len(target_positions_w):
            active_target_pos_w = target_positions_w[active_target_idx]
        else:
            active_target_pos_w = np.full(3, np.nan, dtype=np.float32)
        env_origin = _tensor_env0(env.unwrapped.scene.env_origins)

        mat_log["policy_step"].append(step)
        mat_log["t_policy_s"].append(step * step_dt)
        mat_log["episode_length"].append(_scalar_env0(env.unwrapped.episode_length_buf))
        mat_log["policy_action_raw"].append(_tensor_env0(actions))
        if action_term is not None and hasattr(action_term, "velocity_command"):
            mat_log["velocity_command_b"].append(_tensor_env0(action_term.velocity_command))
        else:
            mat_log["velocity_command_b"].append(np.full(3, np.nan, dtype=np.float32))
        mat_log["reward"].append(_scalar_env0(rewards))
        mat_log["done"].append(bool(_scalar_env0(dones)))

        if robot_asset is not None:
            data = robot_asset.data
            root_pos_w = _tensor_env0(data.root_pos_w)
            root_quat_w = _tensor_env0(data.root_quat_w)
            root_yaw_w = _yaw_from_quat_wxyz(root_quat_w)
            mat_log["root_pos_w"].append(root_pos_w)
            mat_log["root_quat_w"].append(root_quat_w)
            mat_log["root_lin_vel_w"].append(_tensor_env0(data.root_lin_vel_w))
            mat_log["root_ang_vel_w"].append(_tensor_env0(data.root_ang_vel_w))
            mat_log["root_lin_vel_b"].append(_tensor_env0(data.root_lin_vel_b))
            mat_log["root_ang_vel_b"].append(_tensor_env0(data.root_ang_vel_b))
            mat_log["projected_gravity_b"].append(_tensor_env0(data.projected_gravity_b))
            mat_log["joint_pos"].append(_tensor_env0(data.joint_pos))
            mat_log["joint_vel"].append(_tensor_env0(data.joint_vel))
            mat_log["root_yaw_w"].append(root_yaw_w)
            mat_log["root_heading_w"].append(np.array([np.cos(root_yaw_w), np.sin(root_yaw_w), 0.0], dtype=np.float32))
        else:
            for key in (
                "root_pos_w",
                "root_quat_w",
                "root_lin_vel_w",
                "root_ang_vel_w",
                "root_lin_vel_b",
                "root_ang_vel_b",
                "projected_gravity_b",
                "joint_pos",
                "joint_vel",
            ):
                mat_log[key].append(np.array([]))
            mat_log["root_yaw_w"].append(np.nan)
            mat_log["root_heading_w"].append(np.full(3, np.nan, dtype=np.float32))

        if action_term is not None and hasattr(action_term, "processed_actions"):
            mat_log["joint_pos_target"].append(_tensor_env0(action_term.processed_actions))
        else:
            mat_log["joint_pos_target"].append(np.array([]))
        mat_log["cone_red_pos_w"].append(cone_red_pos_w)
        mat_log["cone_red_pos_env"].append(cone_red_pos_w - env_origin)
        mat_log["cone_green_pos_w"].append(cone_green_pos_w)
        mat_log["cone_green_pos_env"].append(cone_green_pos_w - env_origin)
        mat_log["cone_blue_pos_w"].append(cone_blue_pos_w)
        mat_log["cone_blue_pos_env"].append(cone_blue_pos_w - env_origin)
        mat_log["active_target_idx"].append(active_target_idx)
        mat_log["active_target_pos_w"].append(active_target_pos_w)
        mat_log["active_target_pos_env"].append(active_target_pos_w - env_origin)
        mat_log_count += 1
        if csv_logging_enabled and mat_log_count % 50 == 0:
            save_csv_log()

    def write_matlab_plot_script():
        if not args_cli.log_mat:
            return
        script_path = os.path.join(mat_log_dir, args_cli.plot_matlab_script_file)
        mat_file = os.path.basename(mat_log_path)
        script = f"""clear; close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
data = load(fullfile(scriptDir, '{mat_file}'));

t = data.t_policy_s(:);
robotPos = asNx3(data.root_pos_w);
if isfield(data, 'root_yaw_w')
    yaw = data.root_yaw_w(:);
else
    yaw = quatYawWxyz(data.root_quat_w);
end

figure('Name', 'Robot path and targets');
hold on; grid on; axis equal;
plot(robotPos(:, 1), robotPos(:, 2), 'k-', 'LineWidth', 1.8, 'DisplayName', 'robot path');
scatter(robotPos(1, 1), robotPos(1, 2), 70, 'c', 'filled', 'DisplayName', 'start');
scatter(robotPos(end, 1), robotPos(end, 2), 70, 'm', 'filled', 'DisplayName', 'end');

skip = max(1, floor(numel(t) / 30));
idx = 1:skip:numel(t);
quiver(robotPos(idx, 1), robotPos(idx, 2), cos(yaw(idx)), sin(yaw(idx)), 0.25, ...
    'Color', [0.1 0.1 0.1], 'LineWidth', 1.0, 'MaxHeadSize', 1.5, 'DisplayName', 'view direction');

objectFields = ["cone_red_pos_w", "cone_green_pos_w", "cone_blue_pos_w"];
objectLabels = ["red fire extinguisher", "green target", "blue block"];
objectColors = ['r'; 'g'; 'b'];
for i = 1:numel(objectFields)
    field = char(objectFields(i));
    if ~isfield(data, field)
        continue;
    end
    pos = asNx3(data.(field));
    valid = all(isfinite(pos), 2) & pos(:, 3) > -50;
    if ~any(valid)
        continue;
    end
    lastIdx = find(valid, 1, 'last');
    markerPos = pos(lastIdx, :);
    scatter(markerPos(1), markerPos(2), 110, objectColors(i), 'filled', ...
        'DisplayName', char(objectLabels(i)));
    text(markerPos(1), markerPos(2), "  " + objectLabels(i), 'Color', objectColors(i), ...
        'FontWeight', 'bold');
end

if isfield(data, 'active_target_pos_w')
    activePos = asNx3(data.active_target_pos_w);
    valid = all(isfinite(activePos), 2) & activePos(:, 3) > -50;
    if any(valid)
        p = activePos(find(valid, 1, 'last'), :);
        scatter(p(1), p(2), 180, 'p', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y', ...
            'DisplayName', 'active target');
    end
end

xlabel('x world [m]');
ylabel('y world [m]');
title('NaVILA VLA play: robot path, view direction, and objects');
legend('Location', 'bestoutside');
saveas(gcf, fullfile(scriptDir, 'robot_path_targets.png'));

figure('Name', 'Robot state over time');
tiledlayout(3, 1);
nexttile;
plot(t, robotPos(:, 1), 'r', t, robotPos(:, 2), 'g', t, robotPos(:, 3), 'b', 'LineWidth', 1.2);
grid on; ylabel('position [m]'); legend('x', 'y', 'z');
nexttile;
plot(t, yaw, 'k', 'LineWidth', 1.2);
grid on; ylabel('yaw [rad]');
nexttile;
if isfield(data, 'velocity_command_b')
    cmd = asNx3(data.velocity_command_b);
    plot(t, cmd(:, 1), 'r', t, cmd(:, 2), 'g', t, cmd(:, 3), 'b', 'LineWidth', 1.2);
    legend('vx body', 'vy body', 'yaw rate');
end
grid on; xlabel('time [s]'); ylabel('command');
saveas(gcf, fullfile(scriptDir, 'robot_state_timeseries.png'));

function out = asNx3(value)
    out = squeeze(value);
    if isempty(out)
        out = zeros(0, 3);
        return;
    end
    if isvector(out)
        out = reshape(out, 1, []);
    end
    if size(out, 2) < 3 && size(out, 1) >= 3
        out = out';
    end
    out = out(:, 1:3);
end

function yaw = quatYawWxyz(quat)
    q = squeeze(quat);
    if size(q, 2) < 4 && size(q, 1) >= 4
        q = q';
    end
    q = q(:, 1:4);
    w = q(:, 1); x = q(:, 2); y = q(:, 3); z = q(:, 4);
    yaw = atan2(2 .* (w .* z + x .* y), 1 - 2 .* (y .* y + z .* z));
end
"""
        os.makedirs(mat_log_dir, exist_ok=True)
        with open(script_path, "w", newline="\n") as file:
            file.write(script)
        print(f"[INFO]: Wrote MATLAB plot script: {script_path}")

    def save_mat_log():
        nonlocal mat_log_saved
        if not args_cli.log_mat or mat_log_saved:
            return
        mat_log_saved = True
        if mat_log_count == 0:
            print("[WARN]: No MATLAB diagnostics samples were collected.")
            return

        from scipy.io import savemat

        sim_dt = float(getattr(env.unwrapped.cfg.sim, "dt", np.nan))
        decimation = int(getattr(env.unwrapped.cfg, "decimation", 1))
        step_dt = float(getattr(env.unwrapped, "step_dt", sim_dt * decimation))
        low_level_decimation = (
            int(getattr(action_term, "low_level_decimation", 0)) if action_term is not None else 0
        )

        mat_data: dict[str, object] = {
            "task": args_cli.task or "",
            "checkpoint": os.path.basename(resume_path) if resume_path is not None else "",
            "load_run": os.path.basename(log_dir),
            "env_id": mat_log_env_id,
            "sim_dt": sim_dt,
            "policy_dt": step_dt,
            "decimation": decimation,
            "low_level_decimation": low_level_decimation,
            "low_level_dt": sim_dt * low_level_decimation if low_level_decimation > 0 else np.nan,
            "command_labels": np.array(["vx_body", "vy_body", "yaw_rate"], dtype=object),
            "target_labels": np.array(["red fire extinguisher", "green target", "blue block"], dtype=object),
        }
        if action_term is not None and hasattr(action_term, "velocity_range"):
            mat_data["velocity_range"] = action_term.velocity_range.detach().cpu().numpy().copy()
        if robot_asset is not None:
            mat_data["joint_names"] = np.array(robot_asset.joint_names, dtype=object)

        for key, values in mat_log.items():
            if values:
                mat_data[key] = np.asarray(values)

        if action_term is not None and hasattr(action_term, "_diag"):
            diag_name_map = {
                "ll_obs": "ll_obs",
                "ll_output": "ll_output",
                "ll_phys_step": "ll_update_phys_step",
                "q": "phys_joint_pos",
                "dq": "phys_joint_vel",
                "q_des": "phys_joint_pos_target",
                "tau_computed": "phys_tau_computed",
                "tau_applied": "phys_tau_applied",
                "phys_step": "phys_step",
            }
            for src_key, dst_key in diag_name_map.items():
                values = action_term._diag.get(src_key, [])
                if values:
                    mat_data[dst_key] = np.asarray(values)

        savemat(mat_log_path, mat_data, do_compression=True)
        print(f"[INFO]: Saved MATLAB diagnostics: {mat_log_path}")
        write_matlab_plot_script()

    def _sanitize_header(value) -> str:
        return "".join(char if char.isalnum() or char == "_" else "_" for char in str(value))

    def _flat(value) -> list[float]:
        if value is None:
            return []
        array = np.asarray(value)
        if array.size == 0:
            return []
        return array.reshape(-1).tolist()

    def _first_dim(values: list) -> int:
        for value in values:
            size = len(_flat(value))
            if size > 0:
                return size
        return 0

    def _vector_headers(prefix: str, count: int, labels: list[str] | None = None) -> list[str]:
        headers = []
        for idx in range(count):
            suffix = labels[idx] if labels is not None and idx < len(labels) else f"{idx}"
            headers.append(f"{prefix}_{_sanitize_header(suffix)}")
        return headers

    def _append_vector(row: list, value, count: int):
        values = _flat(value)
        values += [np.nan] * max(0, count - len(values))
        row.extend(values[:count])

    def _diag_value(diag: dict, key: str, index: int):
        values = diag.get(key, [])
        if index < len(values):
            return values[index]
        return None

    def _write_csv(path: str, headers: list[str], rows: list[list]):
        with open(path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)

    def save_csv_log():
        if not csv_logging_enabled:
            return
        if mat_log_count == 0:
            print("[WARN]: No CSV diagnostics samples were collected.")
            return

        os.makedirs(csv_log_dir, exist_ok=True)
        sim_dt = float(getattr(env.unwrapped.cfg.sim, "dt", np.nan))
        decimation = int(getattr(env.unwrapped.cfg, "decimation", 1))
        step_dt = float(getattr(env.unwrapped, "step_dt", sim_dt * decimation))
        low_level_decimation = (
            int(getattr(action_term, "low_level_decimation", 0)) if action_term is not None else 0
        )
        joint_names = list(robot_asset.joint_names) if robot_asset is not None else []

        policy_vector_specs = [
            ("policy_action_raw", "action", None),
            ("velocity_command_b", "cmd", ["vx_body", "vy_body", "yaw_rate"]),
            ("root_pos_w", "root_pos_w", ["x", "y", "z"]),
            ("root_quat_w", "root_quat_w", ["w", "x", "y", "z"]),
            ("root_lin_vel_w", "root_lin_vel_w", ["x", "y", "z"]),
            ("root_ang_vel_w", "root_ang_vel_w", ["x", "y", "z"]),
            ("root_lin_vel_b", "root_lin_vel_b", ["x", "y", "z"]),
            ("root_ang_vel_b", "root_ang_vel_b", ["x", "y", "z"]),
            ("projected_gravity_b", "projected_gravity_b", ["x", "y", "z"]),
            ("root_heading_w", "root_heading_w", ["x", "y", "z"]),
            ("cone_red_pos_w", "cone_red_pos_w", ["x", "y", "z"]),
            ("cone_red_pos_env", "cone_red_pos_env", ["x", "y", "z"]),
            ("cone_green_pos_w", "cone_green_pos_w", ["x", "y", "z"]),
            ("cone_green_pos_env", "cone_green_pos_env", ["x", "y", "z"]),
            ("cone_blue_pos_w", "cone_blue_pos_w", ["x", "y", "z"]),
            ("cone_blue_pos_env", "cone_blue_pos_env", ["x", "y", "z"]),
            ("active_target_pos_w", "active_target_pos_w", ["x", "y", "z"]),
            ("active_target_pos_env", "active_target_pos_env", ["x", "y", "z"]),
            ("joint_pos", "joint_pos", joint_names),
            ("joint_vel", "joint_vel", joint_names),
            ("joint_pos_target", "joint_pos_target", joint_names),
        ]
        policy_dims = {key: _first_dim(mat_log[key]) for key, _, _ in policy_vector_specs}
        metadata_headers = [
            "task",
            "checkpoint",
            "load_run",
            "env_id",
            "sim_dt",
            "policy_dt",
            "decimation",
            "low_level_decimation",
            "low_level_dt",
        ]
        metadata_values = [
            args_cli.task or "",
            os.path.basename(resume_path) if resume_path is not None else "",
            os.path.basename(log_dir),
            mat_log_env_id,
            sim_dt,
            step_dt,
            decimation,
            low_level_decimation,
            sim_dt * low_level_decimation if low_level_decimation > 0 else np.nan,
        ]
        policy_headers = metadata_headers + [
            "policy_step",
            "t_policy_s",
            "episode_length",
            "reward",
            "done",
            "root_yaw_w",
            "active_target_idx",
        ]
        for key, prefix, labels in policy_vector_specs:
            policy_headers.extend(_vector_headers(prefix, policy_dims[key], labels))

        policy_rows = []
        for idx in range(mat_log_count):
            row = metadata_values + [
                mat_log["policy_step"][idx],
                mat_log["t_policy_s"][idx],
                mat_log["episode_length"][idx],
                mat_log["reward"][idx],
                int(mat_log["done"][idx]),
                mat_log["root_yaw_w"][idx],
                mat_log["active_target_idx"][idx],
            ]
            for key, _, _ in policy_vector_specs:
                _append_vector(row, mat_log[key][idx], policy_dims[key])
            policy_rows.append(row)
        _write_csv(csv_log_path, policy_headers, policy_rows)
        print(f"[INFO]: Saved CSV diagnostics: {csv_log_path}")

    def save_camera_png(step: int):
        nonlocal camera_png_count, camera_png_warned
        if not args_cli.save_camera_png:
            return
        if camera_png_limit > 0 and camera_png_count >= camera_png_limit:
            return
        if step % camera_png_every != 0:
            return
        sensor = env.unwrapped.scene.sensors.get("front_rgb_camera")
        if sensor is None:
            if not camera_png_warned:
                print("[WARN]: front_rgb_camera sensor not found; camera PNG saving is disabled.")
                camera_png_warned = True
            return
        if "rgb" not in sensor.data.output:
            if not camera_png_warned:
                print("[WARN]: front_rgb_camera has no rgb output yet; waiting for sensor data.")
                camera_png_warned = True
            return

        image = sensor.data.output["rgb"][0]
        if torch.is_tensor(image):
            image = image.detach().cpu()
            if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
                image = image.permute(1, 2, 0)
            image = image.numpy()
        else:
            image = np.asarray(image)
            if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
                image = np.moveaxis(image, 0, -1)

        if image.shape[-1] > 3:
            image = image[..., :3]
        if image.dtype != np.uint8:
            image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
            if image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

        Image.fromarray(image).save(os.path.join(camera_png_dir, f"front_rgb_{camera_png_count:06d}.png"))
        camera_png_count += 1

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    sim_step = 0
    camera_follow_id = 0

    camera_position = (
        env.unwrapped.scene.env_origins[camera_follow_id].cpu().numpy()
        - camera_direction
        + np.array([1.8, -0.8, 0.0])
    )
    env.sim.set_camera_view(camera_position, camera_position + camera_direction)
    # simulate environment
    try:
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                if args_cli.keyboard:
                    actions = env.override_command.clone()
                # env stepping
                obs, rewards, dones, _ = env.step(actions)
                append_mat_log(sim_step, actions, rewards, dones)
                save_camera_png(sim_step)
                sim_step += 1
                if max_play_steps > 0 and sim_step >= max_play_steps:
                    print(f"[INFO]: Reached play limit ({sim_step} steps).")
                    break
            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break
            if MOVE_CAMERA:
                camera_position = env.root_states[camera_follow_id, :3].cpu().numpy() - camera_direction
                env.sim.set_camera_view(camera_position, camera_position + camera_direction)
    finally:
        if vla_policy is not None:
            vla_policy.close()
        save_mat_log()
        save_csv_log()
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
