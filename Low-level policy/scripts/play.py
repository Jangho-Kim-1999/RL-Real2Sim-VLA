# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import math
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

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
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
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--random-command",
    action="store_true",
    default=False,
    help="Use the task's random base velocity command sampler during play instead of the fixed play command.",
)
parser.add_argument(
    "--random-command-resample-s",
    type=float,
    default=None,
    help="Optional fixed resampling time [s] for --random-command.",
)
parser.add_argument(
    "--log-obs-actions",
    action="store_true",
    default=False,
    help="Log observations and actions to a CSV file during play.",
)
parser.add_argument(
    "--log-obs-actions-path",
    type=str,
    default=None,
    help="Optional CSV path for obs/action logging. Defaults under the run log directory.",
)
parser.add_argument(
    "--log-every",
    type=int,
    default=1,
    help="Log every N steps when logging is enabled.",
)
parser.add_argument(
    "--log-max-steps",
    type=int,
    default=None,
    help="Optional maximum number of logged steps.",
)
parser.add_argument(
    "--play-duration-s",
    type=float,
    default=None,
    help="Optional play duration in seconds. Stops after this simulated time.",
)
parser.add_argument(
    "--log-play-signals",
    action="store_true",
    default=False,
    help="MATLAB signal logging is always enabled.",
)
parser.add_argument(
    "--log-play-signals-path",
    type=str,
    default=None,
    help="Optional output path for MATLAB signal CSV. Defaults to logs/data.csv.",
)
parser.add_argument(
    "--log-play-env-id",
    type=int,
    default=0,
    help="(Deprecated) Logged environment is fixed to env_id=0.",
)
parser.add_argument(
    "--dynamic-conversion",
    type=str,
    choices=("on", "off"),
    default="on",
    help="Enable/disable adding tau_comp (dynamic conversion) to applied torque.",
)
parser.add_argument(
    "--tau-compensation-mode",
    type=str,
    choices=("none", "dyn", "prop"),
    default=None,
    help="Select tau_comp formula for play: none=0, dyn=Jm/Bm terms, prop=Jm/Bm+Mlink terms.",
)
parser.add_argument(
    "--play-rand-scope",
    type=str,
    choices=("off", "base", "whole"),
    default="off",
    help="Startup COM/inertia randomization scope during play.",
)
parser.add_argument(
    "--traj",
    type=str,
    choices=("on", "off"),
    default="off",
    help="Legacy switch for scripted trajectory. Use --traj-mode for detailed control.",
)
parser.add_argument(
    "--traj-mode",
    type=str,
    choices=("off", "recip", "circle", "weave"),
    default="off",
    help="Scripted command mode: recip (back-and-forth), circle, or weave.",
)
parser.add_argument("--traj-initial-stop", type=float, default=1.0, help="Initial stop duration [s] before scripted motion.")
# reciprocating trajectory parameters
parser.add_argument("--traj-recip-speed", type=float, default=0.8, help="Recip mode forward/backward speed magnitude [m/s].")
parser.add_argument("--traj-recip-half-period", type=float, default=1.0, help="Recip mode duration [s] per direction.")
parser.add_argument(
    "--traj-recip-cycles",
    type=int,
    default=-1,
    help="Recip mode cycle count (1 cycle = forward+backward). Use negative for infinite repeat.",
)
# circular trajectory parameters
parser.add_argument("--traj-circle-speed", type=float, default=0.8, help="Circle mode linear speed [m/s].")
parser.add_argument("--traj-circle-radius", type=float, default=1.0, help="Circle mode radius [m].")
parser.add_argument(
    "--traj-circle-cycles",
    type=int,
    default=-1,
    help="Circle mode cycle count (1 cycle = one full lap). Use negative for infinite repeat.",
)
parser.add_argument(
    "--traj-circle-direction",
    type=str,
    choices=("ccw", "cw"),
    default="ccw",
    help="Circle mode direction.",
)
# weave trajectory parameters
parser.add_argument("--traj-weave-forward-speed", type=float, default=0.8, help="Weave mode forward speed [m/s].")
parser.add_argument(
    "--traj-weave-lateral-speed",
    type=float,
    default=0.4,
    help="Weave mode sine-path amplitude A [m] (legacy option name kept for compatibility).",
)
parser.add_argument("--traj-weave-period", type=float, default=2.0, help="Weave mode oscillation period [s].")
parser.add_argument(
    "--traj-weave-yaw-gain",
    type=float,
    default=0.8,
    help="Weave mode max yaw rate [rad/s] used to derive theta_amp=max_yaw_rate*T/(2*pi).",
)
# body-pose command override parameters (used when env has commands.body_pose)
parser.add_argument("--body-cmd-z-amp-min", type=float, default=None, help="BodyPose z amplitude min [m].")
parser.add_argument("--body-cmd-z-amp-max", type=float, default=None, help="BodyPose z amplitude max [m].")
parser.add_argument("--body-cmd-z-freq-min", type=float, default=None, help="BodyPose z frequency min [Hz].")
parser.add_argument("--body-cmd-z-freq-max", type=float, default=None, help="BodyPose z frequency max [Hz].")
parser.add_argument("--body-cmd-pitch-amp-min", type=float, default=None, help="BodyPose pitch amplitude min [rad].")
parser.add_argument("--body-cmd-pitch-amp-max", type=float, default=None, help="BodyPose pitch amplitude max [rad].")
parser.add_argument("--body-cmd-pitch-freq-min", type=float, default=None, help="BodyPose pitch frequency min [Hz].")
parser.add_argument("--body-cmd-pitch-freq-max", type=float, default=None, help="BodyPose pitch frequency max [Hz].")
parser.add_argument("--body-cmd-z-bias", type=float, default=None, help="BodyPose z position bias [m].")
parser.add_argument("--body-cmd-pitch-bias", type=float, default=None, help="BodyPose pitch position bias [rad].")
parser.add_argument(
    "--body-cmd-z-mode-prob",
    type=float,
    default=None,
    help="BodyPose z-only mode probability in [0, 1].",
)
parser.add_argument("--body-cmd-resample-min", type=float, default=None, help="BodyPose command resample min time [s].")
parser.add_argument("--body-cmd-resample-max", type=float, default=None, help="BodyPose command resample max time [s].")
parser.add_argument("--attitude-pitch-amp", type=float, default=None, help="Attitude play pitch sine amplitude [rad].")
parser.add_argument("--attitude-pitch-freq", type=float, default=0.5, help="Attitude play pitch sine frequency [Hz].")
parser.add_argument(
    "--attitude-command-duration-s",
    type=float,
    default=None,
    help="Optional active attitude command duration after --traj-initial-stop. Command returns to zero afterward.",
)
parser.add_argument(
    "--attitude-command-cycles",
    type=float,
    default=None,
    help="Optional active attitude sine cycles after --traj-initial-stop. Overrides --attitude-command-duration-s.",
)
parser.add_argument(
    "--hip-armature-scale",
    type=float,
    default=None,
    help="Optional HIP actuator armature scale relative to env_cfg.Jm.",
)
parser.add_argument(
    "--hip-viscous-friction-scale",
    type=float,
    default=None,
    help="Optional HIP actuator viscous_friction scale relative to env_cfg.Bm.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rl_utils import camera_follow

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # Select a tensor observation from possible dict outputs.
    def _select_obs_tensor(obs_any):
        if isinstance(obs_any, dict):
            if "policy" in obs_any:
                return obs_any["policy"]
            for v in obs_any.values():
                if torch.is_tensor(v):
                    return v
        return obs_any

    def _to_numpy(arr):
        if torch.is_tensor(arr):
            return arr.detach().cpu().numpy()
        try:
            import numpy as np

            if isinstance(arr, np.ndarray):
                return arr
        except Exception:
            pass
        return None

    def _sanitize_joint_name(name: str) -> str:
        safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
        return safe.strip("_") or "joint"

    def _quat_wxyz_to_rpy(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = quat_wxyz[:, 0]
        x = quat_wxyz[:, 1]
        y = quat_wxyz[:, 2]
        z = quat_wxyz[:, 3]

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def _to_flat_list(arr: torch.Tensor | None) -> list[float]:
        if arr is None:
            return []
        return arr.detach().cpu().reshape(-1).tolist()

    def _to_numeric_list(arr) -> list[float]:
        if arr is None:
            return []
        if torch.is_tensor(arr):
            return arr.detach().cpu().reshape(-1).tolist()
        if isinstance(arr, (list, tuple)):
            return [float(value) for value in arr]
        return [float(arr)]

    def _fit_length(values: list[float], length: int) -> list[float]:
        if len(values) >= length:
            return values[:length]
        return values + [float("nan")] * (length - len(values))

    def _to_scalar(value) -> float:
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    def _resolve_traj_mode() -> str:
        """Resolve effective scripted trajectory mode with backward compatibility."""
        if args_cli.traj_mode != "off":
            return args_cli.traj_mode
        if args_cli.traj == "on":
            return "recip"
        return "off"

    traj_mode = _resolve_traj_mode()

    def _scripted_base_velocity_command_value(t_s: float) -> tuple[float, float, float]:
        """Return scripted (vx, vy, wz) by selected trajectory mode."""
        if t_s < float(args_cli.traj_initial_stop):
            return 0.0, 0.0, 0.0
        t_active = t_s - float(args_cli.traj_initial_stop)

        if traj_mode == "recip":
            half_period = max(float(args_cli.traj_recip_half_period), 1.0e-6)
            speed = float(args_cli.traj_recip_speed)
            cycle_period = 2.0 * half_period
            cycles = int(args_cli.traj_recip_cycles)
            if cycles >= 0 and t_active >= cycles * cycle_period:
                return 0.0, 0.0, 0.0
            phase_t = t_active % (2.0 * half_period)
            if phase_t < half_period:
                return speed, 0.0, 0.0
            return -speed, 0.0, 0.0

        if traj_mode == "circle":
            vx = float(args_cli.traj_circle_speed)
            radius = max(abs(float(args_cli.traj_circle_radius)), 1.0e-6)
            cycles = int(args_cli.traj_circle_cycles)
            circle_period = 2.0 * math.pi * radius / max(abs(vx), 1.0e-6)
            if cycles >= 0 and t_active >= cycles * circle_period:
                return 0.0, 0.0, 0.0
            direction_sign = 1.0 if args_cli.traj_circle_direction == "ccw" else -1.0
            wz = direction_sign * vx / radius
            return vx, 0.0, wz

        if traj_mode == "weave":
            vx = float(args_cli.traj_weave_forward_speed)
            max_yaw_rate = float(args_cli.traj_weave_yaw_gain)
            T = max(float(args_cli.traj_weave_period), 1.0e-6)
            omega = 2.0 * math.pi / T
            theta_amp = max_yaw_rate * T / (2.0 * math.pi)
            vy = 0.0
            wz = theta_amp * omega * math.cos(omega * t_active)
            return vx, vy, wz

        return 0.0, 0.0, 0.0

    def _scripted_velocity_commands(env) -> torch.Tensor:
        """Return scripted command batch for all envs."""
        t_s = float(env.common_step_counter) * float(env.step_dt)
        vx, vy, wz = _scripted_base_velocity_command_value(t_s)
        cmd = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
        cmd[:, 0] = vx
        cmd[:, 1] = vy
        cmd[:, 2] = wz
        return cmd

    def _attitude_command_duration_s(pitch_freq: float) -> float | None:
        if args_cli.attitude_command_cycles is not None:
            cycles = float(args_cli.attitude_command_cycles)
            freq = float(pitch_freq)
            if cycles <= 0.0:
                raise ValueError("--attitude-command-cycles must be positive.")
            if freq <= 0.0:
                raise ValueError("--attitude-pitch-freq must be positive when --attitude-command-cycles is used.")
            return cycles / freq
        if args_cli.attitude_command_duration_s is not None:
            return float(args_cli.attitude_command_duration_s)
        return None

    def _scripted_attitude_command_value(t_s: float, pitch_amp: float, pitch_freq: float) -> tuple[float, float, float]:
        # command layout for attitude task: [vx_ref, roll_ref, pitch_ref]
        vx_ref = 0.0
        roll_ref = 0.0
        t_active = max(0.0, t_s - float(args_cli.traj_initial_stop))
        attitude_duration_s = _attitude_command_duration_s(float(pitch_freq))
        if attitude_duration_s is not None and t_active >= float(attitude_duration_s):
            return vx_ref, roll_ref, 0.0
        pitch_ref = float(pitch_amp) * math.sin(2.0 * math.pi * float(pitch_freq) * t_active)
        return vx_ref, roll_ref, pitch_ref

    def _scripted_attitude_commands(env, pitch_amp: float, pitch_freq: float) -> torch.Tensor:
        t_s = float(env.common_step_counter) * float(env.step_dt)
        vx_ref, roll_ref, pitch_ref = _scripted_attitude_command_value(t_s, pitch_amp, pitch_freq)
        cmd = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
        cmd[:, 0] = vx_ref
        cmd[:, 1] = roll_ref
        cmd[:, 2] = pitch_ref
        return cmd

    def _collect_play_log_data(base_env, actions_tensor: torch.Tensor, env_id: int):
        if not hasattr(base_env, "scene"):
            return None
        try:
            robot = base_env.scene["robot"]
        except Exception:
            return None
        default_joint_ids = list(range(robot.data.joint_pos.shape[1]))
        default_joint_names = [f"joint_{i}" for i in range(len(default_joint_ids))]

        joint_ids = default_joint_ids
        joint_names = default_joint_names
        joint_pos = robot.data.joint_pos[:, joint_ids]
        joint_vel = robot.data.joint_vel[:, joint_ids]
        joint_acc = robot.data.joint_acc[:, joint_ids]
        joint_vel_raw = joint_vel
        joint_acc_raw = joint_acc
        joint_vel_tustin = None
        joint_acc_tustin = None
        target_joint_pos = joint_pos
        tau_no_comp = None
        tau_comp = None
        applied_torque = robot.data.applied_torque[:, joint_ids]
        base_lin_vel = robot.data.root_lin_vel_b
        base_ang_vel = robot.data.root_ang_vel_b
        base_quat_w = robot.data.root_quat_w
        base_pos_w = robot.data.root_pos_w
        base_command = None
        contact_names = []
        contact_state = None

        if hasattr(base_env, "get_play_log_signals"):
            signals = base_env.get_play_log_signals()
            raw_joint_ids = signals.get("joint_ids", joint_ids)
            if torch.is_tensor(raw_joint_ids):
                joint_ids = raw_joint_ids.tolist()
            else:
                joint_ids = list(raw_joint_ids)
            joint_names = [str(name) for name in signals.get("joint_names", joint_names)]
            joint_pos = signals.get("joint_pos", robot.data.joint_pos[:, joint_ids])
            joint_vel = signals.get("joint_vel", joint_vel)
            joint_acc = signals.get("joint_acc", joint_acc)
            joint_vel_raw = signals.get("joint_vel_raw", joint_vel)
            joint_acc_raw = signals.get("joint_acc_raw", joint_acc)
            joint_vel_tustin = signals.get("joint_vel_tustin", joint_vel_tustin)
            joint_acc_tustin = signals.get("joint_acc_tustin", joint_acc_tustin)
            target_joint_pos = signals.get("target_joint_pos", joint_pos)
            tau_no_comp = signals.get("tau_no_comp", tau_no_comp)
            tau_comp = signals.get("tau_comp", tau_comp)
            applied_torque = signals.get("tau_applied_cmd", applied_torque)
            applied_torque = signals.get("applied_torque", applied_torque)
            base_lin_vel = signals.get("base_lin_vel_b", base_lin_vel)
            base_ang_vel = signals.get("base_ang_vel_b", base_ang_vel)
            base_quat_w = signals.get("base_quat_w", base_quat_w)
            base_pos_w = signals.get("base_pos_w", base_pos_w)
            base_command = signals.get("command", base_command)
            contact_names = [str(name) for name in signals.get("contact_names", contact_names)]
            contact_state = signals.get("contact_state", contact_state)
        else:
            if hasattr(robot, "joint_names"):
                joint_names = [str(name) for name in robot.joint_names]

        command_sources = [base_env, getattr(base_env, "unwrapped", None), getattr(base_env, "_env", None)]
        for source_env in command_sources:
            if source_env is None or not hasattr(source_env, "command_manager"):
                continue
            for command_name in ("attitude", "base_velocity", "body_pose", "jump_mode"):
                try:
                    maybe_cmd = source_env.command_manager.get_command(command_name)
                    if torch.is_tensor(maybe_cmd):
                        base_command = maybe_cmd
                        break
                except Exception:
                    continue
            if torch.is_tensor(base_command):
                break

        # Fallback: direct access to command term object if get_command() path is unavailable.
        if base_command is None:
            for source_env in command_sources:
                cm = getattr(source_env, "command_manager", None)
                terms = getattr(cm, "_terms", None) if cm is not None else None
                if not isinstance(terms, dict):
                    continue
                for command_name in ("attitude", "base_velocity", "body_pose", "jump_mode"):
                    term_obj = terms.get(command_name, None)
                    maybe_cmd = getattr(term_obj, "command", None) if term_obj is not None else None
                    if torch.is_tensor(maybe_cmd):
                        base_command = maybe_cmd
                        break
                if torch.is_tensor(base_command):
                    break

        # Fallback for custom runtime envs that keep current command in a plain attribute.
        if base_command is None:
            for source_env in command_sources:
                if source_env is None:
                    continue
                for attr_name in ("commands", "command", "_commands"):
                    maybe_cmd = getattr(source_env, attr_name, None)
                    if torch.is_tensor(maybe_cmd):
                        base_command = maybe_cmd
                        break
                if torch.is_tensor(base_command):
                    break

        if (
            not torch.is_tensor(base_lin_vel)
            or not torch.is_tensor(base_ang_vel)
            or not torch.is_tensor(base_quat_w)
            or not torch.is_tensor(base_pos_w)
        ):
            return None

        num_envs = int(base_lin_vel.shape[0])
        if num_envs <= 0:
            return None
        env_index = min(max(int(env_id), 0), num_envs - 1)

        if actions_tensor.ndim == 1:
            action_row = actions_tensor
        else:
            action_row = actions_tensor[env_index]

        target_row = target_joint_pos[env_index] if torch.is_tensor(target_joint_pos) else None
        joint_count = int(target_row.numel()) if target_row is not None else len(joint_ids)
        if len(joint_names) != joint_count:
            joint_names = [f"joint_{i}" for i in range(joint_count)]
        joint_names = [_sanitize_joint_name(name) for name in joint_names]

        action_row = action_row.reshape(-1)
        if action_row.numel() == joint_count:
            action_names = joint_names
        else:
            action_names = [f"action_{i}" for i in range(int(action_row.numel()))]

        roll, pitch, yaw = _quat_wxyz_to_rpy(base_quat_w)
        base_lin_vel_row = base_lin_vel[env_index].reshape(-1)
        base_ang_vel_row = base_ang_vel[env_index].reshape(-1)
        if traj_mode != "off":
            t_s = float(base_env.common_step_counter) * float(base_env.step_dt)
            vx, vy, wz = _scripted_base_velocity_command_value(t_s)
            base_command_row = torch.tensor([vx, vy, wz], dtype=torch.float32, device=base_lin_vel.device)
        elif torch.is_tensor(base_command):
            base_command_row = base_command[env_index].reshape(-1)
        else:
            base_command_row = None
        base_speed_xy = torch.linalg.vector_norm(base_lin_vel_row[:2], ord=2)
        base_tilt = torch.sqrt(roll[env_index] ** 2 + pitch[env_index] ** 2)

        return {
            "env_index": env_index,
            "joint_names": joint_names,
            "action_names": action_names,
            "action": action_row,
            "joint_pos": joint_pos[env_index].reshape(-1) if torch.is_tensor(joint_pos) else None,
            "joint_vel": joint_vel[env_index].reshape(-1) if torch.is_tensor(joint_vel) else None,
            "joint_acc": joint_acc[env_index].reshape(-1) if torch.is_tensor(joint_acc) else None,
            "joint_vel_raw": joint_vel_raw[env_index].reshape(-1) if torch.is_tensor(joint_vel_raw) else None,
            "joint_acc_raw": joint_acc_raw[env_index].reshape(-1) if torch.is_tensor(joint_acc_raw) else None,
            "joint_vel_tustin": joint_vel_tustin[env_index].reshape(-1) if torch.is_tensor(joint_vel_tustin) else None,
            "joint_acc_tustin": joint_acc_tustin[env_index].reshape(-1) if torch.is_tensor(joint_acc_tustin) else None,
            "target_joint_pos": target_row.reshape(-1) if target_row is not None else None,
            "tau_no_comp": tau_no_comp[env_index].reshape(-1) if torch.is_tensor(tau_no_comp) else None,
            "tau_comp": tau_comp[env_index].reshape(-1) if torch.is_tensor(tau_comp) else None,
            "applied_torque": applied_torque[env_index].reshape(-1) if torch.is_tensor(applied_torque) else None,
            "base_lin_vel": base_lin_vel_row[:3],
            "base_ang_vel": base_ang_vel_row[:3],
            "command": base_command_row[:3] if base_command_row is not None else None,
            "base_speed_xy": base_speed_xy,
            "base_roll": roll[env_index],
            "base_pitch": pitch[env_index],
            "base_yaw": yaw[env_index],
            "base_tilt": base_tilt,
            "base_pos_x": base_pos_w[env_index, 0],
            "base_pos_y": base_pos_w[env_index, 1],
            "base_pos_z": base_pos_w[env_index, 2],
            "base_height": base_pos_w[env_index, 2],
            "contact_names": [_sanitize_joint_name(name) for name in contact_names],
            "contact_state": contact_state[env_index].reshape(-1) if torch.is_tensor(contact_state) else contact_state,
        }

    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50
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
            "[INFO] Play option applied: --dynamic-conversion {} "
            "(tau_comp {}applied).".format(
                args_cli.dynamic_conversion,
                "" if env_cfg.dynamic_conversion_enable else "not ",
            )
        )
    if args_cli.tau_compensation_mode is not None:
        env_cfg.tau_compensation_mode = args_cli.tau_compensation_mode
        print(f"[INFO] Play option applied: tau_compensation_mode={env_cfg.tau_compensation_mode}.")
    if args_cli.hip_armature_scale is not None or args_cli.hip_viscous_friction_scale is not None:
        try:
            hip_actuator = env_cfg.scene.robot.actuators["HIP"]
            if args_cli.hip_armature_scale is not None:
                hip_actuator.armature = float(getattr(env_cfg, "Jm", 0.0)) * float(args_cli.hip_armature_scale)
            if args_cli.hip_viscous_friction_scale is not None:
                hip_actuator.viscous_friction = float(getattr(env_cfg, "Bm", 0.0)) * float(args_cli.hip_viscous_friction_scale)
            print(
                "[INFO] Play option applied: HIP actuator "
                f"armature={hip_actuator.armature}, viscous_friction={hip_actuator.viscous_friction}."
            )
        except Exception as exc:
            raise RuntimeError("Failed to apply HIP actuator override. Check robot actuator config.") from exc

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.viewer.eye = (2.0, 5.0, 1.0)
    env_cfg.viewer.lookat = (2.0, 0.0, 0.0)
    play_duration_limit_s = float(args_cli.play_duration_s) if args_cli.play_duration_s is not None else None
    if play_duration_limit_s is None and args_cli.attitude_command_cycles is not None:
        play_duration_limit_s = float(args_cli.traj_initial_stop) + float(args_cli.attitude_command_cycles) / float(
            args_cli.attitude_pitch_freq
        )
        print(
            "[INFO] Play option applied: "
            f"play_duration_s auto-set to {play_duration_limit_s:.6f} "
            f"from wait={float(args_cli.traj_initial_stop):.3f}s, cycles={float(args_cli.attitude_command_cycles):.3f}, "
            f"freq={float(args_cli.attitude_pitch_freq):.3f}Hz."
        )
    if play_duration_limit_s is not None:
        play_duration_s = float(play_duration_limit_s)
        sim_dt_cfg = float(getattr(getattr(env_cfg, "sim", None), "dt", 0.0))
        step_dt_cfg = sim_dt_cfg * float(getattr(env_cfg, "decimation", 1))
        min_episode_length_s = play_duration_s + max(step_dt_cfg, 0.1)
        old_episode_length_s = float(getattr(env_cfg, "episode_length_s", min_episode_length_s))
        env_cfg.episode_length_s = max(old_episode_length_s, min_episode_length_s)
        print(
            "[INFO] Play option applied: "
            f"episode_length_s {old_episode_length_s:.3f} -> {env_cfg.episode_length_s:.3f} "
            f"for play_duration_s={play_duration_s:.3f}."
        )

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # Disable stochastic observation corruption during play. Startup COM/inertia
    # randomization can be kept for ablation-matched evaluation via CLI.
    env_cfg.observations.policy.enable_corruption = False
    rand_scope = str(args_cli.play_rand_scope)
    if rand_scope == "base":
        rand_body_names = getattr(env_cfg, "base_link_name", "base_link")
    elif rand_scope == "whole":
        rand_body_names = list(getattr(env_cfg, "whole_link_names", [".*"]))
    else:
        rand_body_names = None

    # Remove unrelated startup/reset/interval randomization so play starts from a
    # deterministic pose while preserving requested COM/inertia randomization.
    env_cfg.events.randomize_rigid_body_material = None
    env_cfg.events.randomize_terrain_material = None
    env_cfg.events.randomize_rigid_body_mass = None
    env_cfg.events.randomize_rigid_body_mass_base = None
    if rand_body_names is None:
        env_cfg.events.randomize_rigid_body_inertia = None
        env_cfg.events.randomize_com_positions = None
        print("[INFO] Play option applied: COM/inertia randomization disabled.")
    else:
        if getattr(env_cfg.events, "randomize_rigid_body_inertia", None) is not None:
            env_cfg.events.randomize_rigid_body_inertia.params["asset_cfg"].body_names = rand_body_names
        if getattr(env_cfg.events, "randomize_com_positions", None) is not None:
            env_cfg.events.randomize_com_positions.params["asset_cfg"].body_names = rand_body_names
        print(f"[INFO] Play option applied: COM/inertia randomization scope={rand_scope}.")
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.randomize_push_robot = None
    env_cfg.curriculum.command_levels = None
    if hasattr(env_cfg, "pd_gain_randomization_enable"):
        env_cfg.pd_gain_randomization_enable = False
    if getattr(env_cfg.events, "randomize_reset_base", None) is not None:
        env_cfg.events.randomize_reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        env_cfg.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
    if getattr(env_cfg.events, "randomize_reset_joints_haa", None) is not None:
        env_cfg.events.randomize_reset_joints_haa.params["position_range"] = (0.0, 0.0)
        env_cfg.events.randomize_reset_joints_haa.params["velocity_range"] = (0.0, 0.0)
    if getattr(env_cfg.events, "randomize_reset_joints_biarticular_hip_knee", None) is not None:
        env_cfg.events.randomize_reset_joints_biarticular_hip_knee.params["qm_range"] = (0.0, 0.0)
        env_cfg.events.randomize_reset_joints_biarticular_hip_knee.params["qb_range"] = (0.0, 0.0)
        env_cfg.events.randomize_reset_joints_biarticular_hip_knee.params["velocity_range"] = (0.0, 0.0)

    if args_cli.keyboard and traj_mode != "off":
        print("[WARN] Scripted trajectory mode is enabled together with --keyboard. Keyboard is ignored.")

    if traj_mode != "off":
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        env_cfg.commands.base_velocity.rel_standing_envs = 1.0
        env_cfg.commands.base_velocity.debug_vis = False
        env_cfg.observations.policy.velocity_commands = ObsTerm(func=_scripted_velocity_commands)
        if traj_mode == "recip":
            recip_cycles_msg = "inf" if int(args_cli.traj_recip_cycles) < 0 else str(int(args_cli.traj_recip_cycles))
            print(
                "[INFO] Scripted traj mode=recip: "
                f"stop={args_cli.traj_initial_stop:.3f}s, speed={args_cli.traj_recip_speed:.3f}m/s, "
                f"half_period={args_cli.traj_recip_half_period:.3f}s, cycles={recip_cycles_msg}."
            )
        elif traj_mode == "circle":
            circle_cycles_msg = "inf" if int(args_cli.traj_circle_cycles) < 0 else str(int(args_cli.traj_circle_cycles))
            print(
                "[INFO] Scripted traj mode=circle: "
                f"stop={args_cli.traj_initial_stop:.3f}s, speed={args_cli.traj_circle_speed:.3f}m/s, "
                f"radius={args_cli.traj_circle_radius:.3f}m, direction={args_cli.traj_circle_direction}, "
                f"cycles={circle_cycles_msg}."
            )
        elif traj_mode == "weave":
            print(
                "[INFO] Scripted traj mode=weave: "
                f"stop={args_cli.traj_initial_stop:.3f}s, vx={args_cli.traj_weave_forward_speed:.3f}m/s, "
                f"max_yaw_rate={args_cli.traj_weave_yaw_gain:.3f}rad/s, period={args_cli.traj_weave_period:.3f}s, "
                "vy=0, theta_amp=max_yaw_rate*T/(2*pi), wz=theta_amp*(2*pi/T)*cos(2*pi*t/T)."
            )
    elif args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1]/2,
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )
    else:
        if getattr(env_cfg.commands, "attitude", None) is not None and getattr(env_cfg.commands, "base_velocity", None) is None:
            if args_cli.attitude_pitch_amp is not None:
                attitude_pitch_amp = float(args_cli.attitude_pitch_amp)
            else:
                try:
                    lo, hi = env_cfg.commands.attitude.ranges.pitch_ref
                    attitude_pitch_amp = float(max(abs(lo), abs(hi)))
                except Exception:
                    attitude_pitch_amp = 0.2
            attitude_pitch_freq = float(args_cli.attitude_pitch_freq)
            env_cfg.observations.policy.velocity_commands = ObsTerm(
                func=lambda env: _scripted_attitude_commands(env, attitude_pitch_amp, attitude_pitch_freq),
            )
            attitude_duration_s = _attitude_command_duration_s(attitude_pitch_freq)
            cycles_msg = (
                f"{float(args_cli.attitude_command_cycles):.3f}"
                if args_cli.attitude_command_cycles is not None
                else "inf"
            )
            print(
                "[INFO] Attitude play command override: "
                f"wait={args_cli.traj_initial_stop:.3f}s, "
                f"duration={attitude_duration_s if attitude_duration_s is not None else 'inf'}s, "
                f"cycles={cycles_msg}, "
                f"roll_ref=0.0, pitch_ref={attitude_pitch_amp:.4f}*sin(2*pi*{attitude_pitch_freq:.4f}*t_active)."
            )

        if getattr(env_cfg.commands, "base_velocity", None) is not None:
            if args_cli.random_command:
                if args_cli.random_command_resample_s is not None:
                    resample_s = max(float(args_cli.random_command_resample_s), 1.0e-6)
                    env_cfg.commands.base_velocity.resampling_time_range = (resample_s, resample_s)
                env_cfg.commands.base_velocity.rel_standing_envs = 0.0
                print(
                    "[INFO] Random base velocity command enabled: "
                    f"lin_vel_x={env_cfg.commands.base_velocity.ranges.lin_vel_x}, "
                    f"lin_vel_y={env_cfg.commands.base_velocity.ranges.lin_vel_y}, "
                    f"ang_vel_z={env_cfg.commands.base_velocity.ranges.ang_vel_z}, "
                    f"resample={env_cfg.commands.base_velocity.resampling_time_range}."
                )
            else:
                env_cfg.commands.base_velocity.ranges.lin_vel_x = (1.5, 1.5)
                env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.5, 0.5)
                env_cfg.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
                env_cfg.commands.base_velocity.rel_standing_envs = 0.0
            env_cfg.commands.base_velocity.debug_vis = False
        if getattr(env_cfg.commands, "body_pose", None) is not None:
            # Body-pose command ranges are configured in env cfg with:
            # z_magnitude/z_frequency_hz and pitch_magnitude/pitch_frequency_hz.
            # Do not write legacy fields (x_ref/pitch_ref), which are unused now.
            pass
        print("[INFO] Play option applied: deterministic reset/command state (no randomization).")

    if getattr(env_cfg.commands, "body_pose", None) is not None:
        body_pose_cmd = env_cfg.commands.body_pose
        body_pose_ranges = body_pose_cmd.ranges

        def _override_range(range_name: str, min_value: float | None, max_value: float | None):
            if min_value is None and max_value is None:
                return
            current_min, current_max = getattr(body_pose_ranges, range_name)
            new_min = current_min if min_value is None else float(min_value)
            new_max = current_max if max_value is None else float(max_value)
            setattr(body_pose_ranges, range_name, (new_min, new_max))

        _override_range("z_magnitude", args_cli.body_cmd_z_amp_min, args_cli.body_cmd_z_amp_max)
        _override_range("z_frequency_hz", args_cli.body_cmd_z_freq_min, args_cli.body_cmd_z_freq_max)
        _override_range("pitch_magnitude", args_cli.body_cmd_pitch_amp_min, args_cli.body_cmd_pitch_amp_max)
        _override_range("pitch_frequency_hz", args_cli.body_cmd_pitch_freq_min, args_cli.body_cmd_pitch_freq_max)

        if args_cli.body_cmd_z_bias is not None and hasattr(body_pose_cmd, "z_pos_bias"):
            body_pose_cmd.z_pos_bias = float(args_cli.body_cmd_z_bias)
        if args_cli.body_cmd_pitch_bias is not None and hasattr(body_pose_cmd, "pitch_pos_bias"):
            body_pose_cmd.pitch_pos_bias = float(args_cli.body_cmd_pitch_bias)
        if args_cli.body_cmd_z_mode_prob is not None and hasattr(body_pose_cmd, "z_mode_probability"):
            body_pose_cmd.z_mode_probability = float(min(max(args_cli.body_cmd_z_mode_prob, 0.0), 1.0))
        if args_cli.body_cmd_resample_min is not None or args_cli.body_cmd_resample_max is not None:
            current_min, current_max = body_pose_cmd.resampling_time_range
            new_min = current_min if args_cli.body_cmd_resample_min is None else float(args_cli.body_cmd_resample_min)
            new_max = current_max if args_cli.body_cmd_resample_max is None else float(args_cli.body_cmd_resample_max)
            body_pose_cmd.resampling_time_range = (new_min, new_max)

        print(
            "[INFO] BodyPose command overrides applied: "
            f"z_amp={body_pose_ranges.z_magnitude}, "
            f"z_freq={body_pose_ranges.z_frequency_hz}, "
            f"pitch_amp={body_pose_ranges.pitch_magnitude}, "
            f"pitch_freq={body_pose_ranges.pitch_frequency_hz}, "
            f"resample={body_pose_cmd.resampling_time_range}, "
            f"z_mode_prob={getattr(body_pose_cmd, 'z_mode_probability', 'n/a')}, "
            f"z_bias={getattr(body_pose_cmd, 'z_pos_bias', 'n/a')}, "
            f"pitch_bias={getattr(body_pose_cmd, 'pitch_pos_bias', 'n/a')}"
        )

    is_body_pose_command_task = (
        getattr(env_cfg.commands, "body_pose", None) is not None
        and getattr(env_cfg.commands, "base_velocity", None) is None
    )
    is_attitude_command_task = (
        getattr(env_cfg.commands, "attitude", None) is not None
        and getattr(env_cfg.commands, "base_velocity", None) is None
    )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(
        policy=policy_nn,
        normalizer=None,
        path=export_model_dir,
        filename="policy.onnx",
    )
    export_policy_as_jit(
        policy=policy_nn,
        normalizer=None,
        path=export_model_dir,
        filename="policy.pt",
    )

    dt = env.unwrapped.step_dt
    # print(dt, "dt")
    # reset environment so the configured joint/root reset state is applied
    obs, _ = env.reset()
    if hasattr(env.unwrapped, "_robot"):
        robot = env.unwrapped._robot
        default_root_quat_w = robot.data.default_root_state[0, 3:7].detach().cpu()
        current_root_quat_w = robot.data.root_quat_w[0].detach().cpu()
        current_roll, current_pitch, current_yaw = _quat_wxyz_to_rpy(current_root_quat_w.unsqueeze(0))
        print(
            "[INFO] Startup orientation env0:"
            f" default_root_quat_w={default_root_quat_w.tolist()},"
            f" current_root_quat_w={current_root_quat_w.tolist()},"
            f" current_roll_deg={float(torch.rad2deg(current_roll)[0]):.3f},"
            f" current_pitch_deg={float(torch.rad2deg(current_pitch)[0]):.3f},"
            f" current_yaw_deg={float(torch.rad2deg(current_yaw)[0]):.3f}"
        )
        if hasattr(env.unwrapped, "_joint_names") and hasattr(env.unwrapped, "_joint_ids"):
            joint_names = list(env.unwrapped._joint_names)
            joint_ids = env.unwrapped._joint_ids
            if torch.is_tensor(joint_ids):
                joint_ids = joint_ids.tolist()
            default_joint_pos = robot.data.default_joint_pos[0, joint_ids].detach().cpu().tolist()
            current_joint_pos = robot.data.joint_pos[0, joint_ids].detach().cpu().tolist()
            print("[INFO] Startup joint pose env0:")
            for joint_name, default_q, current_q in zip(joint_names, default_joint_pos, current_joint_pos):
                print(f"  {joint_name}: default={default_q:.6f}, current={current_q:.6f}")
    from controller import print_leg_foot_linear_jacobian_body_frame_from_sim

    print_leg_foot_linear_jacobian_body_frame_from_sim(env.unwrapped._robot, "FL", env_id=0)
    # setup obs/action CSV logging if enabled
    obs_csv_file = None
    obs_csv_writer = None
    obs_logged_steps = 0
    obs_logged_any = False
    if args_cli.log_obs_actions:
        if args_cli.log_obs_actions_path is not None:
            obs_csv_path = args_cli.log_obs_actions_path
        else:
            obs_csv_path = os.path.join(log_dir, "play_obs_actions.csv")
        obs_csv_dir = os.path.dirname(obs_csv_path)
        if obs_csv_dir:
            os.makedirs(obs_csv_dir, exist_ok=True)
        obs_csv_file = open(obs_csv_path, "w", newline="", buffering=1)
        obs_csv_writer = csv.writer(obs_csv_file)
        print(f"[INFO] Logging observations/actions to: {obs_csv_path}")

    # setup MATLAB-friendly signal logging (always enabled)
    play_csv_file = None
    play_csv_writer = None
    play_logged_steps = 0
    play_logged_any = False
    play_header_written = False
    play_joint_names: list[str] = []
    play_action_names: list[str] = []
    play_contact_names: list[str] = []
    play_grf_names: list[str] = []
    play_fob_leg_names: list[str] = []
    play_force_legs: list[str] = ["FL", "FR", "RL", "RR"]
    play_command_dim = 0
    play_log_env_id = 0
    if hasattr(env.unwrapped, "configure_play_logging"):
        env.unwrapped.configure_play_logging(env_id=play_log_env_id, enable=True)
    if args_cli.log_play_signals_path is not None:
        play_csv_path = args_cli.log_play_signals_path
    else:
        play_csv_path = os.path.join(_REPO_ROOT, "logs", "data.csv")
    play_csv_dir = os.path.dirname(play_csv_path)
    if play_csv_dir:
        os.makedirs(play_csv_dir, exist_ok=True)
    play_csv_file = open(play_csv_path, "w", newline="", buffering=1)
    play_csv_writer = csv.writer(play_csv_file)
    print(f"[INFO] Logging MATLAB signals to: {play_csv_path} (overwrite, env_id={play_log_env_id})")

    timestep = 0
    # simulate environment
    if not simulation_app.is_running():
        print("[WARN] Simulation app is not running. No steps will be logged.")

    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # env stepping
            obs, _, _, _ = env.step(actions)
        timestep += 1

        stop_due_to_log_limit = False

        # CSV logging (per env, per step): observation/action
        if obs_csv_writer is not None and (timestep % max(args_cli.log_every, 1) == 0):
            obs_tensor = _select_obs_tensor(obs)
            obs_np = _to_numpy(obs_tensor)
            act_np = _to_numpy(actions)
            if obs_np is not None and act_np is not None:
                num_envs = obs_np.shape[0]
                obs_dim = obs_np.shape[1] if obs_np.ndim > 1 else 1
                act_dim = act_np.shape[1] if act_np.ndim > 1 else 1

                # write header once
                if obs_csv_file.tell() == 0:
                    header = ["step", "env_id"]
                    header += [f"obs_{i}" for i in range(obs_dim)]
                    header += [f"act_{i}" for i in range(act_dim)]
                    obs_csv_writer.writerow(header)
                    if obs_csv_file is not None:
                        obs_csv_file.flush()

                for env_id in range(num_envs):
                    obs_row = obs_np[env_id].reshape(-1).tolist()
                    act_row = act_np[env_id].reshape(-1).tolist()
                    obs_csv_writer.writerow([timestep, env_id] + obs_row + act_row)

                obs_logged_steps += 1
                obs_logged_any = True
                if obs_csv_file is not None:
                    obs_csv_file.flush()
                if args_cli.log_max_steps is not None and obs_logged_steps >= args_cli.log_max_steps:
                    stop_due_to_log_limit = True

        # CSV logging (single env): joint/base signals for MATLAB
        if play_csv_writer is not None:
            signal_batch = []
            if hasattr(env.unwrapped, "drain_play_log_buffer"):
                signal_batch = env.unwrapped.drain_play_log_buffer()
            if not signal_batch:
                fallback_signal = _collect_play_log_data(env.unwrapped, actions, play_log_env_id)
                if fallback_signal is not None:
                    signal_batch = [fallback_signal]

            if signal_batch:
                if not play_header_written:
                    first_signal = signal_batch[0]
                    play_joint_names = [str(name) for name in first_signal["joint_names"]]
                    play_action_names = [str(name) for name in first_signal.get("action_names", play_joint_names)]
                    play_contact_names = [str(name) for name in first_signal.get("contact_names", [])]
                    play_grf_names = [str(name) for name in first_signal.get("grf_names", [])]
                    play_fob_leg_names = [str(name) for name in first_signal.get("fob_leg_names", [])]
                    header = ["step", "time_s", "policy_step", "env_id"]
                    header += [f"action_{name}" for name in play_action_names]
                    header += [f"target_angle_{name}" for name in play_joint_names]
                    header += [f"joint_angle_{name}" for name in play_joint_names]
                    header += [f"joint_vel_{name}" for name in play_joint_names]
                    header += [f"joint_acc_{name}" for name in play_joint_names]
                    header += [f"joint_vel_raw_{name}" for name in play_joint_names]
                    header += [f"joint_acc_raw_{name}" for name in play_joint_names]
                    header += [f"joint_vel_tustin_{name}" for name in play_joint_names]
                    header += [f"joint_acc_tustin_{name}" for name in play_joint_names]
                    header += [f"tau_no_comp_{name}" for name in play_joint_names]
                    header += [f"tau_comp_{name}" for name in play_joint_names]
                    header += [f"torque_{name}" for name in play_joint_names]
                    header += [f"contact_{name}" for name in play_contact_names]
                    for name in play_grf_names:
                        header += [f"W_grf_x_{name}", f"W_grf_y_{name}", f"W_grf_z_{name}"]
                    for name in play_grf_names:
                        header += [f"W_grf_moment_x_{name}", f"W_grf_moment_y_{name}", f"W_grf_moment_z_{name}"]
                    for name in play_grf_names:
                        header += [
                            f"W_grf_normal_x_{name}",
                            f"W_grf_normal_y_{name}",
                            f"W_grf_normal_z_{name}",
                        ]
                    for name in play_grf_names:
                        header += [
                            f"W_grf_friction_x_{name}",
                            f"W_grf_friction_y_{name}",
                            f"W_grf_friction_z_{name}",
                        ]
                    for name in play_grf_names:
                        header += [
                            f"W_grf_normal_moment_x_{name}",
                            f"W_grf_normal_moment_y_{name}",
                            f"W_grf_normal_moment_z_{name}",
                        ]
                    for name in play_grf_names:
                        header += [
                            f"W_grf_friction_moment_x_{name}",
                            f"W_grf_friction_moment_y_{name}",
                            f"W_grf_friction_moment_z_{name}",
                        ]
                    for name in play_force_legs:
                        header += [f"FOB_residual_tau0_{name}", f"FOB_residual_taum_{name}", f"FOB_residual_taub_{name}"]
                    for name in play_force_legs:
                        header += [f"FOB_force_body_x_{name}", f"FOB_force_body_y_{name}", f"FOB_force_body_z_{name}"]
                    for name in play_force_legs:
                        header += [f"Sensor_force_body_x_{name}", f"Sensor_force_body_y_{name}", f"Sensor_force_body_z_{name}"]
                    header += ["max_contact_body_name", "max_contact_force_norm"]
                    first_command = _to_flat_list(first_signal.get("command"))
                    if traj_mode != "off":
                        command_dim = 3
                    elif is_attitude_command_task:
                        command_dim = 3
                    else:
                        command_dim = len(first_command)
                    if command_dim == 3:
                        if traj_mode == "off" and is_attitude_command_task:
                            command_header = ["cmd_vx_ref", "cmd_roll_ref", "cmd_pitch_ref"]
                        elif traj_mode == "off" and is_body_pose_command_task:
                            command_header = ["cmd_lin_vel_x", "cmd_lin_vel_z", "cmd_ang_vel_pitch"]
                        else:
                            command_header = ["cmd_lin_vel_x", "cmd_lin_vel_y", "cmd_ang_vel_z"]
                    else:
                        command_header = [f"cmd_{i}" for i in range(command_dim)]
                    header += command_header
                    play_command_dim = int(command_dim)
                    header += [
                        "base_lin_vel_x",
                        "base_lin_vel_y",
                        "base_lin_vel_z",
                        "base_ang_vel_x",
                        "base_ang_vel_y",
                        "base_ang_vel_z",
                        "base_roll",
                        "base_pitch",
                        "base_yaw",
                    ]
                    play_csv_writer.writerow(header)
                    if play_csv_file is not None:
                        play_csv_file.flush()
                    play_header_written = True

                for signal_data in signal_batch:
                    action_tensor = signal_data.get("action", actions[signal_data["env_index"]] if actions.ndim > 1 else actions)
                    action_row = _fit_length(_to_flat_list(action_tensor), len(play_action_names))
                    target_row = _fit_length(_to_flat_list(signal_data.get("target_joint_pos")), len(play_joint_names))
                    joint_row = _fit_length(_to_flat_list(signal_data.get("joint_pos")), len(play_joint_names))
                    joint_vel_row = _fit_length(_to_flat_list(signal_data.get("joint_vel")), len(play_joint_names))
                    joint_acc_row = _fit_length(_to_flat_list(signal_data.get("joint_acc")), len(play_joint_names))
                    joint_vel_raw_row = _fit_length(_to_flat_list(signal_data.get("joint_vel_raw")), len(play_joint_names))
                    joint_acc_raw_row = _fit_length(_to_flat_list(signal_data.get("joint_acc_raw")), len(play_joint_names))
                    joint_vel_tustin_row = _fit_length(_to_flat_list(signal_data.get("joint_vel_tustin")), len(play_joint_names))
                    joint_acc_tustin_row = _fit_length(_to_flat_list(signal_data.get("joint_acc_tustin")), len(play_joint_names))
                    tau_no_comp_row = _fit_length(_to_flat_list(signal_data.get("tau_no_comp")), len(play_joint_names))
                    tau_comp_row = _fit_length(_to_flat_list(signal_data.get("tau_comp")), len(play_joint_names))
                    torque_row = _fit_length(_to_flat_list(signal_data.get("applied_torque")), len(play_joint_names))
                    contact_row = _fit_length(_to_numeric_list(signal_data.get("contact_state")), len(play_contact_names))
                    grf_row = _fit_length(_to_flat_list(signal_data.get("W_grf")), len(play_grf_names) * 3)
                    grf_moment_row = _fit_length(
                        _to_flat_list(signal_data.get("W_grf_moment")), len(play_grf_names) * 3
                    )
                    grf_normal_row = _fit_length(_to_flat_list(signal_data.get("W_grf_normal")), len(play_grf_names) * 3)
                    grf_friction_row = _fit_length(
                        _to_flat_list(signal_data.get("W_grf_friction")), len(play_grf_names) * 3
                    )
                    grf_normal_moment_row = _fit_length(
                        _to_flat_list(signal_data.get("W_grf_normal_moment")), len(play_grf_names) * 3
                    )
                    grf_friction_moment_row = _fit_length(
                        _to_flat_list(signal_data.get("W_grf_friction_moment")), len(play_grf_names) * 3
                    )
                    fob_leg_names_row = [str(name) for name in signal_data.get("fob_leg_names", [])]
                    fob_residual_flat = _to_flat_list(signal_data.get("fob_residual"))
                    fob_force_body_flat = _to_flat_list(signal_data.get("fob_force_body"))
                    sensor_force_body_flat = _to_flat_list(signal_data.get("sensor_force_body"))
                    grf_leg_names_row = [str(name) for name in signal_data.get("grf_names", play_grf_names)]

                    def _infer_leg(name: str) -> str | None:
                        upper = str(name).upper()
                        for leg_tag in ("FL", "FR", "RL", "RR"):
                            if leg_tag in upper:
                                return leg_tag
                        return None

                    fob_residual_map: dict[str, list[float]] = {}
                    fob_force_map: dict[str, list[float]] = {}
                    sensor_force_map: dict[str, list[float]] = {}

                    for idx, leg_name in enumerate(fob_leg_names_row):
                        leg_key = _infer_leg(leg_name)
                        if leg_key is None:
                            continue
                        r0 = 3 * idx
                        fob_residual_map[leg_key] = _fit_length(fob_residual_flat[r0 : r0 + 3], 3)
                        fob_force_map[leg_key] = _fit_length(fob_force_body_flat[r0 : r0 + 3], 3)

                    for idx, leg_name in enumerate(grf_leg_names_row):
                        leg_key = _infer_leg(leg_name)
                        if leg_key is None:
                            continue
                        r0 = 3 * idx
                        sensor_values = sensor_force_body_flat[r0 : r0 + 3]
                        if len(sensor_values) == 0 or all(math.isnan(float(v)) for v in sensor_values):
                            sensor_values = grf_row[r0 : r0 + 3]
                        sensor_force_map[leg_key] = _fit_length(sensor_values, 3)
                    # Fallback: if leg tokens were not resolvable but we have 4 feet,
                    # map by canonical order [FL, FR, RL, RR].
                    if len(sensor_force_map) == 0 and len(grf_leg_names_row) == 4:
                        for idx, leg_key in enumerate(play_force_legs):
                            r0 = 3 * idx
                            sensor_values = sensor_force_body_flat[r0 : r0 + 3]
                            if len(sensor_values) == 0 or all(math.isnan(float(v)) for v in sensor_values):
                                sensor_values = grf_row[r0 : r0 + 3]
                            sensor_force_map[leg_key] = _fit_length(sensor_values, 3)

                    fob_residual_row = []
                    fob_force_body_row = []
                    sensor_force_body_row = []
                    for leg_key in play_force_legs:
                        fob_residual_row += fob_residual_map.get(leg_key, [float("nan"), float("nan"), float("nan")])
                        fob_force_body_row += fob_force_map.get(leg_key, [float("nan"), float("nan"), float("nan")])
                        sensor_force_body_row += sensor_force_map.get(leg_key, [float("nan"), float("nan"), float("nan")])
                    if traj_mode != "off":
                        vx, vy, wz = _scripted_base_velocity_command_value(float(signal_data["time_s"]))
                        command_row = [vx, vy, wz]
                    elif is_attitude_command_task:
                        pitch_amp = float(args_cli.attitude_pitch_amp) if args_cli.attitude_pitch_amp is not None else 0.2
                        if args_cli.attitude_pitch_amp is None:
                            try:
                                lo, hi = env_cfg.commands.attitude.ranges.pitch_ref
                                pitch_amp = float(max(abs(lo), abs(hi)))
                            except Exception:
                                pitch_amp = 0.2
                        pitch_freq = float(args_cli.attitude_pitch_freq)
                        vx_ref, roll_ref, pitch_ref = _scripted_attitude_command_value(
                            float(signal_data["time_s"]), pitch_amp, pitch_freq
                        )
                        command_row = [vx_ref, roll_ref, pitch_ref]
                    else:
                        command_row = _to_flat_list(signal_data.get("command"))
                    # Last fallback for attitude task:
                    # policy obs layout is [base_ang_vel(3), projected_gravity(3), velocity_commands(3), ...]
                    if is_attitude_command_task and (len(command_row) == 0 or all(math.isnan(float(x)) for x in command_row)):
                        obs_policy = None
                        if isinstance(obs, dict) and "policy" in obs and torch.is_tensor(obs["policy"]):
                            obs_policy = obs["policy"]
                        elif torch.is_tensor(obs):
                            obs_policy = obs
                        env_index = int(signal_data["env_index"])
                        if torch.is_tensor(obs_policy) and obs_policy.ndim == 2 and obs_policy.shape[1] >= 9:
                            cmd_from_obs = obs_policy[env_index, 6:9].detach().cpu().tolist()
                            command_row = [float(v) for v in cmd_from_obs]
                    if is_attitude_command_task and len(command_row) == 0:
                        command_row = [float("nan"), float("nan"), float("nan")]
                    command_row = _fit_length(command_row, play_command_dim)
                    base_lin_vel_row = _fit_length(_to_flat_list(signal_data.get("base_lin_vel")), 3)
                    base_ang_vel_row = _fit_length(_to_flat_list(signal_data.get("base_ang_vel")), 3)

                    row = [
                        int(signal_data.get("sim_step", timestep)),
                        float(signal_data.get("time_s", timestep * dt)),
                        timestep,
                        int(signal_data["env_index"]),
                    ]
                    row += action_row
                    row += target_row
                    row += joint_row
                    row += joint_vel_row
                    row += joint_acc_row
                    row += joint_vel_raw_row
                    row += joint_acc_raw_row
                    row += joint_vel_tustin_row
                    row += joint_acc_tustin_row
                    row += tau_no_comp_row
                    row += tau_comp_row
                    row += torque_row
                    row += contact_row
                    row += grf_row
                    row += grf_moment_row
                    row += grf_normal_row
                    row += grf_friction_row
                    row += grf_normal_moment_row
                    row += grf_friction_moment_row
                    row += fob_residual_row
                    row += fob_force_body_row
                    row += sensor_force_body_row
                    row += [
                        str(signal_data.get("max_contact_body_name", "")),
                        _to_scalar(signal_data.get("max_contact_force_norm", float("nan"))),
                    ]
                    row += command_row
                    row += base_lin_vel_row
                    row += base_ang_vel_row
                    row += [
                        _to_scalar(signal_data.get("base_roll", float("nan"))),
                        _to_scalar(signal_data.get("base_pitch", float("nan"))),
                        _to_scalar(signal_data.get("base_yaw", float("nan"))),
                    ]
                    play_csv_writer.writerow(row)
                    play_logged_steps += 1
                    play_logged_any = True

                if play_csv_file is not None:
                    play_csv_file.flush()

        if stop_due_to_log_limit:
            break
        if play_duration_limit_s is not None and timestep * dt >= float(play_duration_limit_s):
            break
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if obs_csv_file is not None:
        if not obs_logged_any:
            print("[WARN] No obs/action rows were logged. Check that the sim ran and that logging flags are set.")
        obs_csv_file.close()
    if play_csv_file is not None:
        if not play_logged_any:
            print("[WARN] No MATLAB rows were logged. Check that the sim ran and that logging flags are set.")
        play_csv_file.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()
