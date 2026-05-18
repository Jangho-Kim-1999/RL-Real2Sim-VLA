"""Keyboard teleoperation with random fire-extinguisher and blue-block props.

This script bypasses the high-level policy. Keyboard velocity commands are
converted directly into the raw action expected by the frozen low-level policy.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass

_VRROBO_ISAACLAB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "exts", "vrrobo_isaaclab"))
sys.path.insert(0, os.path.join(_VRROBO_ISAACLAB_DIR, "exts", "rsl_rl"))

from isaaclab.app import AppLauncher


# Edit these values to control spawned object brightness during play.
ENABLE_OBJECT_FILL_LIGHTS = True
FIRE_EXT_LIGHT_INTENSITY_RANGE = (1100.0, 1700.0)
BLUE_BLOCK_LIGHT_INTENSITY_RANGE = (900.0, 1500.0)
FIRE_EXT_LIGHT_COLOR = (1.0, 0.90, 0.78)
BLUE_BLOCK_LIGHT_COLOR = (0.80, 0.88, 1.0)
OBJECT_LIGHT_HEIGHT = 0.85
OBJECT_LIGHT_RADIUS = 0.35
OBJECT_LIGHT_COLOR_JITTER = 0.08


parser = argparse.ArgumentParser(
    description="Keyboard-control MCL robot with randomly spawned fire extinguishers and blue blocks."
)
parser.add_argument("--task", type=str, default="mclquad_seminar_camera_play", help="Task config to use.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Use 1 for keyboard play.")
parser.add_argument("--num_fire_ext", "--fire_count", type=int, default=1, help="Number of fire extinguishers.")
parser.add_argument("--num_blue_blocks", "--blue_count", type=int, default=1, help="Number of blue blocks.")
parser.add_argument("--fire_height", type=float, default=0.50, help="Fire-extinguisher height in meters.")
parser.add_argument("--blue_height", type=float, default=0.6431, help="Blue-block height in meters.")
parser.add_argument("--spawn_x_min", type=float, default=1.2, help="Minimum local X spawn position.")
parser.add_argument("--spawn_x_max", type=float, default=3.0, help="Maximum local X spawn position.")
parser.add_argument("--spawn_y_min", type=float, default=-1.0, help="Minimum local Y spawn position.")
parser.add_argument("--spawn_y_max", type=float, default=1.0, help="Maximum local Y spawn position.")
parser.add_argument("--min_robot_dist", type=float, default=0.65, help="Minimum XY distance from robot spawn.")
parser.add_argument("--min_object_dist", type=float, default=0.30, help="Extra XY spacing between spawned objects.")
parser.add_argument("--object_seed", type=int, default=7, help="Random seed for object placement.")
parser.add_argument(
    "--disable_object_lights",
    action="store_true",
    default=not ENABLE_OBJECT_FILL_LIGHTS,
    help="Do not add fill lights above spawned objects.",
)
parser.add_argument(
    "--object_light_intensity_min",
    type=float,
    default=None,
    help="Optional CLI override for minimum per-object fill light intensity.",
)
parser.add_argument(
    "--object_light_intensity_max",
    type=float,
    default=None,
    help="Optional CLI override for maximum per-object fill light intensity.",
)
parser.add_argument(
    "--object_light_height",
    type=float,
    default=OBJECT_LIGHT_HEIGHT,
    help="Height above each object for its fill light.",
)
parser.add_argument(
    "--object_light_radius",
    type=float,
    default=OBJECT_LIGHT_RADIUS,
    help="Radius of each per-object fill light.",
)
parser.add_argument(
    "--object_light_color_jitter",
    type=float,
    default=OBJECT_LIGHT_COLOR_JITTER,
    help="Random fill-light color jitter amount.",
)
parser.add_argument("--keep_task_target", action="store_true", help="Keep the task's built-in target object randomization.")
parser.add_argument("--keep_camera_obs", action="store_true", help="Keep policy camera observations enabled.")
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
import isaacsim.core.utils.prims as prim_utils
import omni.appwindow
import torch
from pxr import Gf

import vrrobo_isaaclab.tasks  # noqa: F401
from carb.input import KeyboardEventType
from isaaclab_tasks.utils import parse_env_cfg
from vrrobo_isaaclab.tasks.vrrobo.config.mclquad import mclquad_env_cfg as mcl_cfg


@dataclass(frozen=True)
class RandomObjectSpec:
    name: str
    usd_path: str
    count: int
    height: float
    original_height: float
    bottom_to_origin: float
    original_radius: float
    light_intensity_range: tuple[float, float]
    light_color: tuple[float, float, float]

    @property
    def scale(self) -> float:
        return self.height / self.original_height

    @property
    def z_offset(self) -> float:
        return self.bottom_to_origin * self.scale

    @property
    def radius(self) -> float:
        return self.original_radius * self.scale


def resolve_asset_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_VRROBO_ISAACLAB_DIR, path))


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = 0.5 * yaw
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def command_to_raw_action(command: torch.Tensor, velocity_range: torch.Tensor) -> torch.Tensor:
    normalized = torch.clamp(command / velocity_range, -0.95, 0.95)
    return 0.5 * torch.log((1.0 + normalized) / (1.0 - normalized))


def configure_keyboard_env(env_cfg) -> None:
    """Remove high-level-only sensors/target motion that are unnecessary for keyboard play."""
    if not args_cli.keep_task_target and hasattr(env_cfg.commands, "rgb_command"):
        env_cfg.commands.rgb_command.target_asset_names = None
        env_cfg.commands.rgb_command.target_pose_range = None
        env_cfg.commands.rgb_command.target_position_offsets = None

    if args_cli.keep_camera_obs:
        return

    policy_obs = getattr(env_cfg.observations, "policy", None)
    if policy_obs is not None:
        for term_name in ("camera_image", "gs_image"):
            if hasattr(policy_obs, term_name):
                setattr(policy_obs, term_name, None)
    if hasattr(env_cfg.scene, "front_rgb_camera"):
        env_cfg.scene.front_rgb_camera = None


def build_object_specs() -> list[RandomObjectSpec]:
    return [
        RandomObjectSpec(
            name="fire_ext",
            usd_path=resolve_asset_path(mcl_cfg.FIRE_EXT_USD),
            count=max(0, args_cli.num_fire_ext),
            height=max(0.01, args_cli.fire_height),
            original_height=mcl_cfg.FIRE_EXT_ORIGINAL_HEIGHT,
            bottom_to_origin=mcl_cfg.FIRE_EXT_BOTTOM_TO_ORIGIN,
            original_radius=0.35,
            light_intensity_range=FIRE_EXT_LIGHT_INTENSITY_RANGE,
            light_color=FIRE_EXT_LIGHT_COLOR,
        ),
        RandomObjectSpec(
            name="blue_block",
            usd_path=resolve_asset_path(mcl_cfg.BLUE_BLOCK_USD),
            count=max(0, args_cli.num_blue_blocks),
            height=max(0.01, args_cli.blue_height),
            original_height=mcl_cfg.BLUE_BLOCK_ORIGINAL_HEIGHT,
            bottom_to_origin=mcl_cfg.BLUE_BLOCK_BOTTOM_TO_ORIGIN,
            original_radius=0.56,
            light_intensity_range=BLUE_BLOCK_LIGHT_INTENSITY_RANGE,
            light_color=BLUE_BLOCK_LIGHT_COLOR,
        ),
    ]


def sample_object_poses(
    specs: list[RandomObjectSpec],
    rng: random.Random,
    robot_xy: tuple[float, float],
) -> list[tuple[RandomObjectSpec, int, float, float, float]]:
    placed: list[tuple[float, float, float]] = []
    sampled: list[tuple[RandomObjectSpec, int, float, float, float]] = []
    x_range = (min(args_cli.spawn_x_min, args_cli.spawn_x_max), max(args_cli.spawn_x_min, args_cli.spawn_x_max))
    y_range = (min(args_cli.spawn_y_min, args_cli.spawn_y_max), max(args_cli.spawn_y_min, args_cli.spawn_y_max))

    for spec in specs:
        for obj_idx in range(spec.count):
            accepted = False
            x = y = yaw = 0.0
            for _ in range(2000):
                x = rng.uniform(*x_range)
                y = rng.uniform(*y_range)
                yaw = rng.uniform(-math.pi, math.pi)
                if math.hypot(x - robot_xy[0], y - robot_xy[1]) < args_cli.min_robot_dist + spec.radius:
                    continue
                if any(
                    math.hypot(x - other_x, y - other_y)
                    < args_cli.min_object_dist + spec.radius + other_radius
                    for other_x, other_y, other_radius in placed
                ):
                    continue
                accepted = True
                break
            if not accepted:
                print(f"[WARN] Dense spawn area: placing {spec.name}_{obj_idx} at last sampled pose.")
            placed.append((x, y, spec.radius))
            sampled.append((spec, obj_idx, x, y, yaw))
    return sampled


def sample_light_intensity(spec: RandomObjectSpec, rng: random.Random) -> float:
    range_min, range_max = spec.light_intensity_range
    if args_cli.object_light_intensity_min is not None:
        range_min = args_cli.object_light_intensity_min
    if args_cli.object_light_intensity_max is not None:
        range_max = args_cli.object_light_intensity_max
    light_min = max(0.0, min(range_min, range_max))
    light_max = max(0.0, max(range_min, range_max))
    if light_min == light_max:
        return light_min
    return rng.uniform(light_min, light_max)


def sample_light_color(spec: RandomObjectSpec, rng: random.Random) -> Gf.Vec3f:
    jitter = max(0.0, args_cli.object_light_color_jitter)
    base_color = spec.light_color
    color = [min(1.0, max(0.0, channel + rng.uniform(-jitter, jitter))) for channel in base_color]
    return Gf.Vec3f(*color)


def create_object_fill_light(root_path: str, spec: RandomObjectSpec, obj_idx: int, x: float, y: float, rng: random.Random):
    if args_cli.disable_object_lights:
        return None
    intensity = sample_light_intensity(spec, rng)
    light_path = f"{root_path}/Lights/{spec.name}_{obj_idx}_fill_light"
    prim_utils.create_prim(
        prim_path=light_path,
        prim_type="SphereLight",
        translation=(x, y, spec.z_offset + max(0.0, args_cli.object_light_height)),
        attributes={
            "inputs:intensity": intensity,
            "inputs:radius": max(0.01, args_cli.object_light_radius),
            "inputs:color": sample_light_color(spec, rng),
        },
    )
    return intensity


def recreate_random_objects(base_env, robot, specs: list[RandomObjectSpec], rng: random.Random) -> None:
    total_count = sum(spec.count for spec in specs)
    for env_id in range(base_env.num_envs):
        root_path = f"/World/envs/env_{env_id}/KeyboardRandomObjects"
        if prim_utils.is_prim_path_valid(root_path):
            prim_utils.delete_prim(root_path)
        prim_utils.create_prim(root_path, prim_type="Xform")
        prim_utils.create_prim(f"{root_path}/Lights", prim_type="Xform")

        env_origin = base_env.scene.env_origins[env_id]
        robot_xy = (
            float((robot.data.root_pos_w[env_id, 0] - env_origin[0]).item()),
            float((robot.data.root_pos_w[env_id, 1] - env_origin[1]).item()),
        )
        sampled = sample_object_poses(specs, rng, robot_xy)
        sampled_light_intensities: list[float | None] = []
        for spec, obj_idx, x, y, yaw in sampled:
            prim_utils.create_prim(
                prim_path=f"{root_path}/{spec.name}_{obj_idx}",
                prim_type="Xform",
                translation=(x, y, spec.z_offset),
                orientation=yaw_to_quat_wxyz(yaw),
                scale=(spec.scale, spec.scale, spec.scale),
                usd_path=spec.usd_path,
            )
            sampled_light_intensities.append(create_object_fill_light(root_path, spec, obj_idx, x, y, rng))
        if env_id == 0:
            print(f"[INFO] Spawned {total_count} random objects under {root_path}")
            for (spec, obj_idx, x, y, yaw), light_intensity in zip(sampled, sampled_light_intensities):
                light_msg = "off" if light_intensity is None else f"{light_intensity:.0f}"
                print(
                    f"       {spec.name}_{obj_idx}: x={x:.2f}, y={y:.2f}, "
                    f"z={spec.z_offset:.2f}, yaw={yaw:.2f}, scale={spec.scale:.3f}, light={light_msg}"
                )


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    configure_keyboard_env(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    obs, _ = env.reset()
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    foot_body_ids, _ = robot.find_bodies(".*_foot", preserve_order=True)
    action_term = base_env.action_manager.get_term("joint_pos")
    velocity_range = action_term.velocity_range
    rng = random.Random(args_cli.object_seed)
    object_specs = build_object_specs()
    recreate_random_objects(base_env, robot, object_specs, rng)
    base_env.sim.render()

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

    print("[INFO] Keyboard random-object play")
    print("       W/S: vx +/- | A/D: vy +/- | F/G: yaw +/- | X: stop | R: reset+respawn | ESC: quit")
    print(f"[INFO] task={args_cli.task}, envs={base_env.num_envs}")
    print(f"[INFO] low_level_policy={env_cfg.actions.joint_pos.policy_dir}")
    print(f"[INFO] high_level_policy=disabled")
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
                    recreate_random_objects(base_env, robot, object_specs, rng)
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
                obs, _, terminated, truncated, _ = env.step(raw_action)

                if step % print_every == 0:
                    measured = torch.cat([robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], dim=1)
                    base_z = robot.data.root_pos_w[0, 2]
                    min_foot_z = robot.data.body_link_pos_w[0, foot_body_ids, 2].amin()
                    print(
                        f"[{step:06d}] cmd={command[0].detach().cpu().numpy()} "
                        f"measured={measured[0].detach().cpu().numpy()} "
                        f"base_z={float(base_z):.3f} min_foot_z={float(min_foot_z):.3f}"
                    )

                if bool((terminated | truncated).any()):
                    obs, _ = env.reset()
                    recreate_random_objects(base_env, robot, object_specs, rng)
                    base_env.sim.render()

                step += 1
    finally:
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
