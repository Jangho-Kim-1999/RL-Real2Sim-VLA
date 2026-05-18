"""Keyboard teleoperation for the frozen low-level policy with GS image capture."""

from __future__ import annotations

import argparse
import os
import shutil
import sys

_VRROBO_ISAACLAB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "exts", "vrrobo_isaaclab"))
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "exts", "rsl_rl"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Keyboard-control MCL robot and save GS-rendered observations as PNGs.")
parser.add_argument("--task", type=str, default="mclquad_gs_play", help="Task config to use.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Use 1 for keyboard play.")
parser.add_argument("--capture_dir", type=str, default="logs/keyboard_gs_capture", help="Directory for PNG captures.")
parser.add_argument("--capture_interval_s", type=float, default=1.0, help="Seconds between saved GS PNGs.")
parser.add_argument("--capture_limit", type=int, default=0, help="Max PNGs to save. 0 means unlimited.")
parser.add_argument("--clear_capture_dir", action="store_true", help="Delete capture_dir before starting.")
parser.add_argument("--max_steps", type=int, default=0, help="Exit after this many env steps. 0 means run until closed.")
parser.add_argument("--vx", type=float, default=0.8, help="Forward/backward command speed for W/S.")
parser.add_argument("--vy", type=float, default=0.4, help="Lateral command speed for A/D.")
parser.add_argument("--wz", type=float, default=0.8, help="Yaw command speed for F/G.")
parser.add_argument("--print_interval_s", type=float, default=1.0, help="Seconds between console status prints.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# A visible app window is needed for keyboard events. Do not pass --headless for this script.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb.input
import gymnasium as gym
import omni.appwindow
import torch

import vrrobo_isaaclab.tasks  # noqa: F401
from carb.input import KeyboardEventType
from isaaclab_tasks.utils import parse_env_cfg


def command_to_raw_action(command: torch.Tensor, velocity_range: torch.Tensor) -> torch.Tensor:
    normalized = torch.clamp(command / velocity_range, -0.95, 0.95)
    return 0.5 * torch.log((1.0 + normalized) / (1.0 - normalized))


def configure_capture(env_cfg) -> str:
    capture_dir = args_cli.capture_dir
    if not os.path.isabs(capture_dir):
        capture_dir = os.path.abspath(os.path.join(_VRROBO_ISAACLAB_DIR, capture_dir))
    if args_cli.clear_capture_dir and os.path.isdir(capture_dir):
        shutil.rmtree(capture_dir)
    os.makedirs(capture_dir, exist_ok=True)

    step_dt = float(env_cfg.decimation * env_cfg.sim.dt)
    capture_every = max(1, round(args_cli.capture_interval_s / step_dt))
    os.environ["VRROBO_GS_DEBUG_DIR"] = capture_dir
    os.environ["VRROBO_GS_DEBUG_LIMIT"] = str(args_cli.capture_limit)
    os.environ["VRROBO_GS_DEBUG_EVERY"] = str(capture_every)
    os.environ["VRROBO_GS_DEBUG_SKIP_BLACK"] = "1"
    print(f"[INFO] capture_dir={capture_dir}")
    print(f"[INFO] env_step_dt={step_dt:.3f}s, capture_every={capture_every} env steps")
    return capture_dir


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    capture_dir = configure_capture(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    obs, _ = env.reset()
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    action_term = base_env.action_manager.get_term("joint_pos")
    velocity_range = action_term.velocity_range

    pressed: set[carb.input.KeyboardInput] = set()
    should_reset = False
    should_stop = False

    def on_keyboard_input(event):
        nonlocal should_reset, should_stop
        if event.type in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
            pressed.add(event.input)
            if event.input == carb.input.KeyboardInput.X:
                pressed.clear()
            elif event.input == carb.input.KeyboardInput.R:
                should_reset = True
            elif event.input == carb.input.KeyboardInput.ESCAPE:
                should_stop = True
        elif event.type == KeyboardEventType.KEY_RELEASE:
            pressed.discard(event.input)

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_interface = carb.input.acquire_input_interface()
    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_input)

    camera_direction = torch.tensor([5.0, 0.0, -6.0], device=base_env.device)
    camera_position = base_env.scene.env_origins[0] - camera_direction + torch.tensor(
        [1.8, -0.8, 0.0], device=base_env.device
    )
    base_env.sim.set_camera_view(camera_position.cpu().numpy(), (camera_position + camera_direction).cpu().numpy())

    print("[INFO] Keyboard controls")
    print("       W/S: vx +/- | A/D: vy +/- | F/G: yaw +/- | X: stop | R: reset | ESC: quit")
    print(f"[INFO] velocity_range={velocity_range.detach().cpu().tolist()}")
    print("[INFO] Keep render_server.py running in another terminal. Do not run another client on port 12345.")

    command = torch.zeros(base_env.num_envs, 3, device=base_env.device)
    print_every = max(1, round(args_cli.print_interval_s / float(base_env.step_dt)))
    step = 0

    with torch.inference_mode():
        while simulation_app.is_running() and not should_stop:
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break
            if should_reset:
                obs, _ = env.reset()
                should_reset = False

            command.zero_()
            if carb.input.KeyboardInput.W in pressed:
                command[:, 0] += args_cli.vx
            if carb.input.KeyboardInput.S in pressed:
                command[:, 0] -= args_cli.vx
            if carb.input.KeyboardInput.A in pressed:
                command[:, 1] += args_cli.vy
            if carb.input.KeyboardInput.D in pressed:
                command[:, 1] -= args_cli.vy
            if carb.input.KeyboardInput.F in pressed:
                command[:, 2] += args_cli.wz
            if carb.input.KeyboardInput.G in pressed:
                command[:, 2] -= args_cli.wz

            raw_action = command_to_raw_action(command, velocity_range)
            obs, _, terminated, truncated, _ = env.step(raw_action)

            if step % print_every == 0:
                measured = torch.cat([robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], dim=1)
                print(
                    f"[{step:06d}] cmd={command[0].detach().cpu().numpy()} "
                    f"measured={measured[0].detach().cpu().numpy()} "
                    f"captures={capture_dir}"
                )

            if bool((terminated | truncated).any()):
                obs, _ = env.reset()

            step += 1

    input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
