from __future__ import annotations

import os
import torch
import torch.nn as nn
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.string as string_utils
import omni.log
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.modules.actor_critic import get_activation

from . import actions_cfg


class JointAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: actions_cfg.JointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency only when order does not matter.
        # If preserve_order=True, replacing resolved ids with slice(None) can silently
        # switch to the USD/native joint order and break policies trained with a fixed
        # joint order.
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self._num_joints, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class JointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


class VelocityCommandAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    """The configuration of the action term."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset

        policy_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.policy_dir))
        policy_action_dim = cfg.policy_action_dim if cfg.policy_action_dim is not None else self._num_joints
        if cfg.load_actor_only:
            self.policy_model = self._load_actor_only_policy(policy_dir, cfg.policy_obs_dim, policy_action_dim)
            self.policy = self.policy_model
        else:
            policy_cfg = {
                "class_name": cfg.policy_class_name,
                "init_noise_std": cfg.policy_init_noise_std,
                "actor_hidden_dims": cfg.policy_actor_hidden_dims,
                "critic_hidden_dims": cfg.policy_critic_hidden_dims,
                "activation": cfg.policy_activation,
            }

            actor_critic_class = eval(policy_cfg["class_name"])  # ActorCritic
            actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
                cfg.policy_obs_dim, cfg.policy_critic_obs_dim, policy_action_dim, **policy_cfg
            ).to(self.device)
            actor_critic.load_state_dict(self._load_checkpoint(policy_dir)["model_state_dict"])
            actor_critic.eval()
            self.policy_model = actor_critic
            self.policy = actor_critic.act_inference
        self.velocity_range = torch.tensor(cfg.velocity_range, device=self.device)
        self.velocity_command = torch.zeros([self.num_envs, 3], device=self.device)
        self.uplevel_frequency = cfg.uplevel_frequency
        self.low_level_decimation = cfg.low_level_decimation
        self.velocity_command_start_idx = cfg.velocity_command_start_idx
        self.lowlevel_counter = 0
        self.tanh = torch.nn.Tanh()
        self.use_biarticular_torque_control = cfg.use_biarticular_torque_control
        self.pd_kp = float(cfg.pd_kp)
        self.pd_kd = float(cfg.pd_kd)
        self.torque_limit = cfg.torque_limit
        self.torque_delay_steps = max(0, int(cfg.torque_delay_steps))
        self._joint_id_list = self._resolve_joint_id_list()
        self._default_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._hip_knee_pairs = self._resolve_hip_knee_pairs()
        self._torque_queue = (
            torch.zeros(self.num_envs, self.torque_delay_steps + 1, self._num_joints, device=self.device)
            if self.torque_delay_steps > 0
            else None
        )

        # Diagnostics — populated only when _diag_enabled = True (env index 0 only)
        self._diag_enabled = False
        self._diag: dict = {
            "ll_obs": [], "ll_output": [], "ll_phys_step": [],
            "q": [], "dq": [], "q_des": [],
            "tau_computed": [], "tau_applied": [], "phys_step": [],
        }
        self._diag_phys_step = 0

    def _load_actor_only_policy(self, policy_dir: str, obs_dim: int, action_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        hidden_dims = list(self.cfg.policy_actor_hidden_dims)
        activation_name = self.cfg.policy_activation
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(get_activation(activation_name))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        actor = nn.Sequential(*layers).to(self.device)

        checkpoint = self._load_checkpoint(policy_dir)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        actor_state = {key[len("actor.") :]: value for key, value in state_dict.items() if key.startswith("actor.")}
        if not actor_state:
            raise ValueError(f"No actor.* keys found in low-level policy checkpoint: {policy_dir}")
        actor.load_state_dict(actor_state, strict=True)
        actor.eval()
        for param in actor.parameters():
            param.requires_grad_(False)
        return actor

    def _load_checkpoint(self, policy_dir: str):
        try:
            return torch.load(policy_dir, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(policy_dir, map_location=self.device)

    def _resolve_joint_id_list(self) -> list[int]:
        if isinstance(self._joint_ids, slice):
            return list(range(self._asset.num_joints))
        if torch.is_tensor(self._joint_ids):
            return [int(idx) for idx in self._joint_ids.tolist()]
        return [int(idx) for idx in self._joint_ids]

    def _resolve_hip_knee_pairs(self) -> list[tuple[int, int]]:
        hip_token = str(getattr(self._env.cfg, "hip_joint_token", "HIP")).upper()
        knee_token = str(getattr(self._env.cfg, "knee_joint_token", "KNEE")).upper()

        knee_by_prefix: dict[str, int] = {}
        for local_idx, name in enumerate(self._joint_names):
            upper_name = name.upper()
            knee_pos = upper_name.find(knee_token)
            if knee_pos >= 0:
                knee_by_prefix[name[:knee_pos]] = local_idx

        pairs: list[tuple[int, int]] = []
        for local_idx, name in enumerate(self._joint_names):
            upper_name = name.upper()
            hip_pos = upper_name.find(hip_token)
            if hip_pos < 0:
                continue
            knee_idx = knee_by_prefix.get(name[:hip_pos])
            if knee_idx is not None:
                pairs.append((local_idx, knee_idx))
        pairs.sort(key=lambda pair: pair[0])
        return pairs

    def process_actions(self, actions: torch.Tensor):
        self.velocity_command = self.tanh(actions) * self.velocity_range

    def apply_actions(self):
        ll_ran = (self.lowlevel_counter % self.low_level_decimation == 0)
        if ll_ran:
            obs = self.get_observations()
            with torch.inference_mode():
                start = self.velocity_command_start_idx
                obs[:, start : start + 3] = self.velocity_command
                joint_positions = self.policy(obs)
            super().process_actions(joint_positions)
            if self._diag_enabled:
                self._diag["ll_obs"].append(obs[0].detach().cpu().numpy().copy())
                self._diag["ll_output"].append(joint_positions[0].detach().cpu().numpy().copy())
                self._diag["ll_phys_step"].append(self._diag_phys_step)
        self.lowlevel_counter += 1
        self.lowlevel_counter %= self.low_level_decimation
        if self._diag_enabled:
            self._diag["q"].append(self._asset.data.joint_pos[0, self._joint_ids].detach().cpu().numpy().copy())
            self._diag["dq"].append(self._asset.data.joint_vel[0, self._joint_ids].detach().cpu().numpy().copy())
            self._diag["q_des"].append(self.processed_actions[0].detach().cpu().numpy().copy())
            self._diag["phys_step"].append(self._diag_phys_step)
            self._diag_phys_step += 1
        if self.use_biarticular_torque_control:
            self._apply_biarticular_torque_control()
        else:
            self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def _apply_biarticular_torque_control(self):
        if not self._hip_knee_pairs:
            raise RuntimeError(
                "Biarticular torque control is enabled, but no HIP/KNEE pairs were resolved. "
                "Check joint names and hip_joint_token/knee_joint_token."
            )

        q = self._asset.data.joint_pos[:, self._joint_ids]
        dq = self._asset.data.joint_vel[:, self._joint_ids]
        q_des = self.processed_actions
        tau = torch.zeros_like(q)

        hip_ids = [pair[0] for pair in self._hip_knee_pairs]
        knee_ids = [pair[1] for pair in self._hip_knee_pairs]

        action_delta = q_des - self._default_joint_pos
        qm_des = self._default_joint_pos[:, hip_ids] + action_delta[:, hip_ids]
        qb_des = self._default_joint_pos[:, hip_ids] + self._default_joint_pos[:, knee_ids] + action_delta[:, knee_ids]

        qm = q[:, hip_ids]
        qb = q[:, hip_ids] + q[:, knee_ids]
        dqm = dq[:, hip_ids]
        dqb = dq[:, hip_ids] + dq[:, knee_ids]

        tau_bi = self.pd_kp * torch.stack((qm_des - qm, qb_des - qb), dim=-1) - self.pd_kd * torch.stack(
            (dqm, dqb), dim=-1
        )
        tau_m = tau_bi[..., 0]
        tau_b = tau_bi[..., 1]
        tau[:, hip_ids] = tau_m + tau_b
        tau[:, knee_ids] = tau_b

        paired_ids = set(hip_ids + knee_ids)
        remaining_ids = [idx for idx in range(self._num_joints) if idx not in paired_ids]
        if remaining_ids:
            tau[:, remaining_ids] = self.pd_kp * (q_des[:, remaining_ids] - q[:, remaining_ids]) - self.pd_kd * dq[
                :, remaining_ids
            ]

        if self.torque_limit is not None:
            tau = torch.clamp(tau, -float(self.torque_limit), float(self.torque_limit))

        tau_computed = tau.clone()

        if self._torque_queue is not None:
            self._torque_queue[:, :-1, :] = self._torque_queue[:, 1:, :].clone()
            self._torque_queue[:, -1, :] = tau
            tau = self._torque_queue[:, 0, :]

        if self._diag_enabled:
            self._diag["tau_computed"].append(tau_computed[0].detach().cpu().numpy().copy())
            self._diag["tau_applied"].append(tau[0].detach().cpu().numpy().copy())

        self._asset.set_joint_effort_target(tau, joint_ids=self._joint_ids)

    def get_observations(self):
        """Returns the current observations of the environment."""
        obs = self._env.observation_manager.compute_group("locomotion")
        return obs

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if self._torque_queue is not None:
            self._torque_queue[env_ids] = 0.0

    @property
    def action_dim(self) -> int:
        return 3
