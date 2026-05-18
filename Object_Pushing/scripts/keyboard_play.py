"""Keyboard teleoperation for the object pushing scene.

This script bypasses high-level policy learning/inference. Keyboard velocity
commands are sent directly to the frozen low-level locomotion policy through
the same action term used by the object-pushing training environment.
"""

from __future__ import annotations

import argparse
import os
import sys

_OBJECT_PUSHING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WORKSPACE_DIR = os.path.abspath(os.path.join(_OBJECT_PUSHING_DIR, ".."))
_VRROBO_ISAACLAB_DIR = os.path.join(_WORKSPACE_DIR, "VR-Robo-main", "vrrobo_isaaclab")
sys.path.insert(0, _OBJECT_PUSHING_DIR)
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "exts", "vrrobo_isaaclab"))
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "exts", "rsl_rl"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Keyboard-control the robot in the object pushing scene.")
parser.add_argument("--task", type=str, default="mclquad_object_pushing_play", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Use 1 for keyboard play.")
parser.add_argument("--max_steps", type=int, default=0, help="Exit after this many high-level steps. 0 means no limit.")
parser.add_argument("--vx", type=float, default=0.8, help="Forward/backward command speed for W/S.")
parser.add_argument("--vy", type=float, default=0.4, help="Lateral command speed for A/D.")
parser.add_argument("--wz", type=float, default=0.8, help="Yaw command speed for F/G.")
parser.add_argument("--print_interval_s", type=float, default=1.0, help="Seconds between console status prints.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# A visible app window is needed for keyboard events. Do not run this with --headless.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb.input
import gymnasium as gym
import omni.appwindow
import torch
from carb.input import KeyboardEventType
from isaaclab_tasks.utils import parse_env_cfg

import object_pushing  # noqa: F401


def command_to_raw_action(command: torch.Tensor, velocity_range: torch.Tensor) -> torch.Tensor:
    normalized = torch.clamp(command / velocity_range, -0.95, 0.95)
    return 0.5 * torch.log((1.0 + normalized) / (1.0 - normalized))


def object_linear_velocity(push_object) -> torch.Tensor:
    if hasattr(push_object.data, "root_lin_vel_w"):
        return push_object.data.root_lin_vel_w
    return push_object.data.root_vel_w[:, :3]


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    obs, _ = env.reset()
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    push_object = base_env.scene["push_object"]
    target = base_env.scene["target_marker"]
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

    camera_direction = torch.tensor([4.5, -2.0, -5.0], device=base_env.device)
    camera_position = base_env.scene.env_origins[0] - camera_direction + torch.tensor(
        [0.8, 0.0, 0.0], device=base_env.device
    )
    base_env.sim.set_camera_view(camera_position.cpu().numpy(), (camera_position + camera_direction).cpu().numpy())

    print("[INFO] Object pushing keyboard play")
    print("       W/S: vx +/- | A/D: vy +/- | F/G: yaw +/- | X: stop | R: reset | ESC: quit")
    print(f"[INFO] task={args_cli.task}, envs={base_env.num_envs}")
    print(f"[INFO] low_level_policy={env_cfg.actions.joint_pos.policy_dir}")
    print("[INFO] high_level_policy=disabled")
    print(f"[INFO] velocity_range={velocity_range.detach().cpu().tolist()}")

    command = torch.zeros(base_env.num_envs, 3, device=base_env.device)
    print_every = max(1, round(args_cli.print_interval_s / float(base_env.step_dt)))
    step = 0

    try:
        with torch.inference_mode():
            while simulation_app.is_running() and not should_stop:
                if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                    break
                if should_reset:
                    obs, _ = env.reset()
                    base_env.sim.render()
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
                obs, rewards, terminated, truncated, _ = env.step(raw_action)

                if step % print_every == 0:
                    object_pos = push_object.data.root_pos_w[0]
                    target_pos = target.data.root_pos_w[0]
                    robot_pos = robot.data.root_pos_w[0]
                    object_vel = object_linear_velocity(push_object)[0]
                    object_target_dist = torch.norm((object_pos - target_pos)[:2])
                    robot_object_dist = torch.norm((robot_pos - object_pos)[:2])
                    print(
                        f"[{step:06d}] cmd={command[0].detach().cpu().numpy()} "
                        f"reward={float(rewards[0]):.3f} "
                        f"robot-object={float(robot_object_dist):.3f}m "
                        f"object-target={float(object_target_dist):.3f}m "
                        f"object_xy=({float(object_pos[0]):.3f}, {float(object_pos[1]):.3f}) "
                        f"object_vxy=({float(object_vel[0]):.3f}, {float(object_vel[1]):.3f})"
                    )

                if bool((terminated | truncated).any()):
                    obs, _ = env.reset()
                    base_env.sim.render()

                step += 1
    finally:
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
