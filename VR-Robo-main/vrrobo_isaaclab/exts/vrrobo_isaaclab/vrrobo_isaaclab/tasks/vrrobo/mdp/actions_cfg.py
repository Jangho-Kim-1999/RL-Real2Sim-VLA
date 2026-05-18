from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import actions


@configclass
class JointActionCfg(ActionTermCfg):
    """Configuration for the base joint action term.

    See :class:`JointAction` for more details.
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""
    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""


@configclass
class JointPositionActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = actions.JointPositionAction

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """


@configclass
class VelocityCommandActionCfg(JointPositionActionCfg):
    """Configuration for the velocity command action term.

    See :class:`VelocityCommandAction` for more details.
    """

    class_type: type[ActionTerm] = actions.VelocityCommandAction

    use_default_offset: bool = True

    velocity_range: list = MISSING

    policy_dir: str = MISSING

    uplevel_frequency: int = MISSING

    low_level_decimation: int = 4

    policy_class_name: str = "ActorCriticRecurrent"

    policy_obs_dim: int = 48

    policy_critic_obs_dim: int = 235

    policy_action_dim: int | None = None

    policy_actor_hidden_dims: list[int] = [512, 256, 128]

    policy_critic_hidden_dims: list[int] = [512, 256, 128]

    policy_activation: str = "elu"

    policy_init_noise_std: float = 0.75

    load_actor_only: bool = False

    velocity_command_start_idx: int = 9

    use_biarticular_torque_control: bool = False

    pd_kp: float = 50.0

    pd_kd: float = 1.5

    torque_limit: float | None = 60.0

    torque_delay_steps: int = 0
