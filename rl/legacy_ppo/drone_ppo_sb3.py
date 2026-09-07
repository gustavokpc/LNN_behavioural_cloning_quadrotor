import argparse
import time
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import gym as legacy_gym
import numpy as np
import torch as th
from ncps.torch import CfC, LTC
from ncps.wirings import NCP
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RecurrentRolloutBufferSamples
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from .quadcopter_animation import animation # type: ignore
from .quadcopter_envs import Quadcopter3DGates, RANDOMIZATION # type: ignore
from .quadcopter_hover_envs import Quadcopter3DHover # type: ignore


LEGACY_RL_ROOT = Path(__file__).resolve().parents[1]
LEGACY_RL_OUTPUT_ROOT = LEGACY_RL_ROOT.parent / "organized_plots" / "rl_runs"


# ---------------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------------

class GradientEpisodePrintCallback(BaseCallback):
    """Prints rollout reward and average gradient norm diagnostics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self._rollout_idx = 0
        self._episode_grad_norms: list[float] = []
        self._step_rewards: list[float] = []
        self._episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self._step_rewards.extend(np.asarray(rewards, dtype=np.float64).reshape(-1).tolist())

        infos = self.locals.get("infos") or []
        for info in infos:
            episode_info = info.get("episode") if isinstance(info, dict) else None
            if episode_info is not None and "r" in episode_info:
                self._episode_rewards.append(float(episode_info["r"]))

        if dones is None:
            return True
        for done in dones:
            if not done:
                continue
            grad_norm = getattr(self.model.policy, "_last_grad_norm", None)
            if grad_norm is None:
                continue
            self._episode_grad_norms.append(float(grad_norm))
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_idx += 1
        parts = [f"Rollout {self._rollout_idx}"]
        if self._step_rewards:
            parts.append(f"avg_step_reward={float(np.mean(self._step_rewards)):.6f}")
        if self._episode_rewards:
            parts.append(f"avg_episode_reward={float(np.mean(self._episode_rewards)):.3f}")
            parts.append(f"episodes={len(self._episode_rewards)}")
        if self._episode_grad_norms:
            parts.append(f"avg_episode_grad_norm={float(np.mean(self._episode_grad_norms)):.6f}")
        else:
            parts.append("no episode grad norms collected")
        print(" | ".join(parts))
        self._episode_grad_norms.clear()
        self._step_rewards.clear()
        self._episode_rewards.clear()


def attach_gradient_logger(policy: th.nn.Module) -> None:
    """Wrap optimizer.step to store the latest grad norm on the policy."""
    optimizer = getattr(policy, "optimizer", None)
    if optimizer is None:
        raise AttributeError("Policy has no optimizer; cannot attach gradient logger.")
    if getattr(optimizer, "_grad_logging_wrapped", False):
        return

    original_step = optimizer.step

    def step_with_logging(*args, **kwargs):
        total_norm_sq = 0.0
        for param in policy.parameters():
            if param.grad is None:
                continue
            param_norm = param.grad.detach().data.norm(2).item()
            total_norm_sq += param_norm ** 2
        policy._last_grad_norm = total_norm_sq ** 0.5
        policy._grad_step = getattr(policy, "_grad_step", 0) + 1
        return original_step(*args, **kwargs)

    optimizer.step = step_with_logging
    optimizer._grad_logging_wrapped = True


# ---------------------------------------------------------------------------
# Saliency helpers used during rendering/evaluation
# ---------------------------------------------------------------------------

def _init_recurrent_state(policy: RecurrentActorCriticPolicy, n_envs: int) -> tuple[np.ndarray, ...]:
    """Create zero hidden/cell states in the format expected by sb3-contrib."""
    state = np.concatenate([np.zeros(policy.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1)
    return (state, state)


def _compute_action_saliency(policy: th.nn.Module, obs: np.ndarray) -> np.ndarray:
    """Return |d log pi(a|obs) / d obs| for the feedforward policy."""
    policy.set_training_mode(False)
    obs_tensor, _ = policy.obs_to_tensor(obs)
    obs_tensor = obs_tensor.detach().clone().requires_grad_(True)
    policy.zero_grad(set_to_none=True)
    with th.enable_grad():
        dist = policy.get_distribution(obs_tensor)
        action = dist.get_actions(deterministic=False).detach()
        log_prob = dist.log_prob(action)
        log_prob.sum().backward()
    grad = obs_tensor.grad.detach()
    return grad.view(grad.shape[0], -1).abs().cpu().numpy()


def _compute_action_saliency_recurrent(
    policy: RecurrentActorCriticPolicy,
    obs: np.ndarray,
    state: tuple[np.ndarray, ...] | None,
    episode_starts: np.ndarray,
) -> np.ndarray:
    """Return |d log pi(a|obs) / d obs| while preserving recurrent state inputs."""
    policy.set_training_mode(False)
    obs_tensor, _ = policy.obs_to_tensor(obs)
    obs_tensor = obs_tensor.detach().clone().requires_grad_(True)
    n_envs = obs_tensor.shape[0]
    if state is None:
        state = _init_recurrent_state(policy, n_envs)
    states_t = (
        th.tensor(state[0], dtype=th.float32, device=policy.device),
        th.tensor(state[1], dtype=th.float32, device=policy.device),
    )
    episode_starts_t = th.tensor(episode_starts, dtype=th.float32, device=policy.device)
    policy.zero_grad(set_to_none=True)
    with th.enable_grad():
        dist, _ = policy.get_distribution(obs_tensor, states_t, episode_starts_t)
        actions = dist.get_actions(deterministic=False).detach()
        log_prob = dist.log_prob(actions)
        log_prob.sum().backward()
    grad = obs_tensor.grad.detach()
    return grad.view(grad.shape[0], -1).abs().cpu().numpy()


# ---------------------------------------------------------------------------
# Policy definitions
# ---------------------------------------------------------------------------

class MyCfCFeaturesExtractor(BaseFeaturesExtractor):
    """
    CNN feature extractor for CfC policies.

    :param observation_space: Observation space
    :param features_dim: Number of features to extract
    """

    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 512):
        super(MyCfCFeaturesExtractor, self).__init__(observation_space, features_dim)
        # Treat the full observation vector as a 1D signal with a single channel.
        self._conv_in_channels = 1
        self.cnn = th.nn.Sequential(
            th.nn.Conv1d(self._conv_in_channels, 64, kernel_size=5, stride=2, padding=2),
            th.nn.ReLU(inplace=True),
            th.nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            th.nn.ReLU(inplace=True),
            th.nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2),
            th.nn.ReLU(inplace=True),
            th.nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            th.nn.ReLU(inplace=True),
            th.nn.AdaptiveAvgPool1d(1),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample = np.asarray(observation_space.sample(), dtype=np.float32)
            sample_tensor = th.from_numpy(sample).view(1, self._conv_in_channels, -1)
            n_flatten = self.cnn(sample_tensor).shape[1]

        # self.linear = th.nn.Sequential(th.nn.Linear(n_flatten, features_dim), th.nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # VecEnv observations arrive as (batch, obs_dim); Conv1d expects
        # (batch, channels, sequence_length).
        if observations.ndim == 2:
            observations = observations.unsqueeze(1)
        else:
            observations = observations.view(observations.shape[0], self._conv_in_channels, -1)
        features = self.cnn(observations).squeeze(-1)
        # return self.linear(features)
        return features


class LogStdClampMixin:
    """Caps Gaussian policy log_std during distribution construction and optimizer updates."""

    max_log_std: float | None

    def _get_capped_log_std(self, log_std: th.Tensor) -> th.Tensor:
        if self.max_log_std is None:
            return log_std
        return th.clamp(log_std, max=self.max_log_std)

    def _clip_log_std_parameter_(self) -> None:
        if self.max_log_std is None:
            return
        log_std = getattr(self, "log_std", None)
        if log_std is None:
            return
        with th.no_grad():
            log_std.clamp_(max=self.max_log_std)

    def _wrap_optimizer_step_with_log_std_cap(self) -> None:
        optimizer = getattr(self, "optimizer", None)
        if optimizer is None or getattr(optimizer, "_log_std_cap_wrapped", False):
            return

        original_step = optimizer.step

        def step_with_log_std_cap(*args, **kwargs):
            # Clamp after every optimizer update so exploration variance cannot
            # drift above the requested cap during long training runs.
            out = original_step(*args, **kwargs)
            self._clip_log_std_parameter_()
            return out

        optimizer.step = step_with_log_std_cap
        optimizer._log_std_cap_wrapped = True

    def _get_action_dist_from_latent(
        self,
        latent_pi: th.Tensor,
        latent_sde: Optional[th.Tensor] = None,
    ):
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(
                mean_actions,
                self._get_capped_log_std(self.log_std),
            )

        if isinstance(self.action_dist, StateDependentNoiseDistribution):
            if latent_sde is None:
                latent_sde = latent_pi
            return self.action_dist.proba_distribution(
                mean_actions,
                self._get_capped_log_std(self.log_std),
                latent_sde,
            )

        return super()._get_action_dist_from_latent(latent_pi, latent_sde)


class ClampedLogStdActorCriticPolicy(LogStdClampMixin, ActorCriticPolicy):
    """Standard PPO policy with the same log_std safety cap used by the CfC policy."""

    def __init__(self, *args, max_log_std: float | None = 1.0, **kwargs):
        self.max_log_std = max_log_std
        super().__init__(*args, **kwargs)
        self._clip_log_std_parameter_()
        self._wrap_optimizer_step_with_log_std_cap()


def _shape_cfc_timespans(
    cfc_timespan: float | th.Tensor,
    batch_size: int,
    hidden_size: int,
    reference: th.Tensor,
) -> th.Tensor:
    """Broadcast a scalar/vector timespan to the shape required by ncps CfC."""
    if not th.is_tensor(cfc_timespan):
        cfc_timespan = th.as_tensor(
            cfc_timespan,
            device=reference.device,
            dtype=reference.dtype,
        )

    if cfc_timespan.dim() == 0:
        return cfc_timespan.view(1, 1, 1).expand(batch_size, 1, hidden_size)

    if cfc_timespan.dim() == 1:
        if cfc_timespan.shape[0] == 1:
            return cfc_timespan.view(1, 1, 1).expand(batch_size, 1, hidden_size)
        if cfc_timespan.shape[0] == batch_size:
            return cfc_timespan.view(batch_size, 1, 1).expand(batch_size, 1, hidden_size)
        if cfc_timespan.shape[0] == hidden_size:
            return cfc_timespan.view(1, 1, hidden_size).expand(batch_size, 1, hidden_size)
        raise ValueError("cfc_timespan 1D tensor must have length 1, batch size, or hidden size.")

    if cfc_timespan.dim() == 2:
        if cfc_timespan.shape == (batch_size, 1):
            return cfc_timespan.view(batch_size, 1, 1).expand(batch_size, 1, hidden_size)
        if cfc_timespan.shape == (batch_size, hidden_size):
            return cfc_timespan.unsqueeze(1)
        raise ValueError("cfc_timespan 2D tensor must be (batch, 1) or (batch, hidden).")

    if cfc_timespan.dim() == 3:
        if cfc_timespan.shape == (batch_size, 1, hidden_size):
            return cfc_timespan
        raise ValueError("cfc_timespan 3D tensor must be (batch, 1, hidden).")

    raise ValueError("cfc_timespan must be a scalar or a tensor with up to 3 dims.")


class RecurrentActorCriticCfCPolicy(LogStdClampMixin, RecurrentActorCriticPolicy):
    """RecurrentActorCriticPolicy variant that swaps SB3's internal LSTMs for CfC/LTC blocks."""

    recurrent_cell_cls = CfC

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        *args,
        cfc_timespan: float | th.Tensor | None = None,
        cfc_kwargs: dict[str, Any] | None = None,
        ncp_kwargs: dict[str, int] | None = None,
        max_log_std: float | None = 1.0,
        **kwargs,
    ):
        # Keep this optional for ordinary CfC/LTC and older checkpoints; wired
        # NCP policies use it when constructing their recurrent cell.
        self._cfc_lr_schedule = lr_schedule
        self._cfc_kwargs = cfc_kwargs or {}
        self._ncp_kwargs = ncp_kwargs
        self._cfc_timespan = cfc_timespan
        self.max_log_std = max_log_std
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            *args,
            **kwargs,
        )
        # CfC/LTC keeps a single hidden state (no cell), so we only track one layer.
        self.lstm_actor = self._make_recurrent_cell(
            self.features_dim,
        )
        recurrent_state_size = int(getattr(self.lstm_actor, "state_size", self.lstm_output_dim))
        self.lstm_hidden_state_shape = (1, 1, recurrent_state_size)
        self.lstm_actor.cfc_timespan = self._cfc_timespan
        for name, param in self.lstm_actor.named_parameters():
            if "bias" in name:
                th.nn.init.constant_(param, 0)
            elif "weight" in name:
                th.nn.init.orthogonal_(param, 1.0)
        self.lstm_actor.hidden_size = recurrent_state_size
        self.lstm_actor.num_layers = 1

        if self.enable_critic_lstm:
            self.lstm_critic = self._make_recurrent_cell(
                self.features_dim,
            )
            self.lstm_critic.cfc_timespan = self._cfc_timespan
            self.lstm_critic.hidden_size = int(getattr(self.lstm_critic, "state_size", self.lstm_output_dim))
            self.lstm_critic.num_layers = 1
        elif self.shared_lstm:
            self.lstm_critic = None
        else:
            self.lstm_critic = None
            hidden_dim = self.lstm_output_dim
            self.critic = th.nn.Sequential(
                th.nn.Linear(self.features_dim, hidden_dim),
                th.nn.ReLU(),
                th.nn.Linear(hidden_dim, hidden_dim),
                th.nn.ReLU(),
                th.nn.Linear(hidden_dim, hidden_dim),
                th.nn.ReLU(),
            )

        # Rebuild optimizer so it tracks the CfC parameters instead of the discarded LSTMs.
        self.optimizer = self.optimizer_class(self.parameters(), lr=self._cfc_lr_schedule(1), **self.optimizer_kwargs)
        self._clip_log_std_parameter_()
        self._wrap_optimizer_step_with_log_std_cap()

    def _make_recurrent_cell(self, input_size: int):
        units: int | NCP = self.lstm_output_dim
        if self._ncp_kwargs is not None:
            units = NCP(motor_neurons=self.lstm_output_dim, **self._ncp_kwargs)
        return self.recurrent_cell_cls(
            input_size, units, batch_first=True, return_sequences=True, **self._cfc_kwargs
        )

    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        rnn_module: th.nn.Module,
    ):
        if isinstance(rnn_module, (CfC, LTC)):
            hidden, cell = lstm_states
            hidden = hidden.squeeze(0)
            if hidden.dim() == 1:
                hidden = hidden.unsqueeze(0)

            # sb3-contrib flattens rollout data as (time * envs, features).
            # CfC processes a single time step at a time here so we can reset
            # hidden state exactly where each environment starts a new episode.
            n_seq = hidden.shape[0]
            seq_features = features.reshape((n_seq, -1, rnn_module.input_size)).swapaxes(0, 1)
            seq_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
            cfc_timespan = getattr(rnn_module, "cfc_timespan", None)
            hidden_size = getattr(rnn_module, "state_size", None)
            if hidden_size is None:
                hidden_size = getattr(rnn_module, "hidden_size", None)

            outputs = []
            for step_features, step_start in zip(seq_features, seq_starts, strict=True):
                hidden = (1.0 - step_start).unsqueeze(-1) * hidden
                if cfc_timespan is None:
                    step_out, hidden = rnn_module(step_features.unsqueeze(1), hidden)
                elif getattr(rnn_module, "state_size", None) != getattr(rnn_module, "output_size", None):
                    # ncps' wired cell passes one elapsed-time value through
                    # layers of different widths. Run each environment with a
                    # scalar timespan to avoid broadcasting it as a state vector.
                    sample_outputs = []
                    sample_hidden = []
                    for sample_features, sample_state in zip(step_features, hidden, strict=True):
                        sample_timespan = th.as_tensor(
                            cfc_timespan,
                            device=sample_features.device,
                            dtype=sample_features.dtype,
                        ).reshape(1, 1, 1)
                        out, state = rnn_module(
                            sample_features.reshape(1, 1, -1),
                            sample_state.reshape(1, -1),
                            timespans=sample_timespan,
                        )
                        sample_outputs.append(out)
                        sample_hidden.append(state)
                    step_out = th.cat(sample_outputs, dim=0)
                    hidden = th.cat(sample_hidden, dim=0)
                else:
                    if hidden_size is None:
                        raise RuntimeError("Unable to resolve CfC hidden size for timespan shaping.")
                    step_timespans = _shape_cfc_timespans(
                        cfc_timespan,
                        batch_size=step_features.shape[0],
                        hidden_size=hidden_size,
                        reference=step_features,
                    )
                    step_features = step_features.unsqueeze(1)
                    step_out, hidden = rnn_module(step_features, hidden, timespans=step_timespans)
                outputs.append(step_out[:, -1, :])

            stacked = th.stack(outputs).swapaxes(0, 1)
            flattened = th.flatten(stacked, start_dim=0, end_dim=1)
            return flattened, (hidden.unsqueeze(0), cell)

        return RecurrentActorCriticPolicy._process_sequence(
            features=features,
            lstm_states=lstm_states,
            episode_starts=episode_starts,
            lstm=rnn_module,
        )


class RecurrentActorCriticLTCPolicy(RecurrentActorCriticCfCPolicy):
    """RecurrentActorCriticPolicy variant that swaps SB3's internal LSTMs for LTC blocks."""

    recurrent_cell_cls = LTC


class RecurrentActorCriticNCPCfCPolicy(RecurrentActorCriticCfCPolicy):
    """CfC recurrent policy with an NCP sparse wiring."""


def resolve_algorithm(args) -> tuple[type[PPO] | type[RecurrentPPO], type[ActorCriticPolicy], dict[str, Any]]:
    """Map CLI policy choice to the SB3 algorithm, policy class, and kwargs."""
    ncp_scale_factor = float(getattr(args, "ncp_scale_factor", 1.0))
    if ncp_scale_factor <= 0.0:
        raise ValueError("--ncp-scale-factor must be positive.")

    def scaled_ncp(value: int, minimum: int = 1) -> int:
        return max(minimum, int(round(value * ncp_scale_factor)))

    shared_cfc_kwargs = dict(
        mixed_memory=False,
        # mode="default",
    )
    if args.policy_type == "ppo":
        policy_kwargs = dict(
            net_arch=dict(activation_fn=th.nn.ReLU, pi=[64, 64, 64], vf=[64, 64, 64], log_std_init=0.0),
            max_log_std=args.max_log_std,
        )
        return PPO, ClampedLogStdActorCriticPolicy, policy_kwargs
    elif args.policy_type in {"recurrent_ppo", "recurrent_ppo_ltc", "recurrent_ppo_ncp_cfc"}:
        algo_cls = RecurrentPPO
        features_extractor_cls = (
            FlattenExtractor if args.use_flatten_features else MyCfCFeaturesExtractor
        )
        common_policy_kwargs: dict[str, Any] = dict(
            features_extractor_class=features_extractor_cls,
            share_features_extractor=True,
            normalize_images=False,
        )
        policy_kwargs = dict(
            **common_policy_kwargs,
            shared_lstm=False,
            enable_critic_lstm=False,
            lstm_hidden_size=args.cell_size,
            cfc_timespan=args.cfc_timespan,
            cfc_kwargs=shared_cfc_kwargs,
            ncp_kwargs=(
                dict(
                    inter_neurons=scaled_ncp(args.ncp_inter_neurons),
                    command_neurons=scaled_ncp(args.ncp_command_neurons),
                    sensory_fanout=scaled_ncp(args.ncp_sensory_fanout),
                    inter_fanout=scaled_ncp(args.ncp_inter_fanout),
                    recurrent_command_synapses=scaled_ncp(args.ncp_recurrent_command_synapses, minimum=0),
                    motor_fanin=scaled_ncp(args.ncp_motor_fanin),
                )
                if args.policy_type == "recurrent_ppo_ncp_cfc" else None
            ),
            max_log_std=args.max_log_std,
        )
        policy_class = {
            "recurrent_ppo_ltc": RecurrentActorCriticLTCPolicy,
            "recurrent_ppo_ncp_cfc": RecurrentActorCriticNCPCfCPolicy,
        }.get(args.policy_type, RecurrentActorCriticCfCPolicy)
        return algo_cls, policy_class, policy_kwargs

    raise ValueError(f"Unsupported policy type: {args.policy_type}")


# ---------------------------------------------------------------------------
# Gym compatibility and run modes
# ---------------------------------------------------------------------------


class LegacyGymAdapter(gym.Env):
    """Wraps classic Gym envs so they satisfy the Gymnasium Env interface."""

    metadata = {}

    def __init__(self, legacy_env):
        super().__init__()
        self.legacy_env = legacy_env
        self.observation_space = self._convert_space(legacy_env.observation_space)
        self.action_space = self._convert_space(legacy_env.action_space)
        self.reward_range = getattr(legacy_env, "reward_range", (-float("inf"), float("inf")))
        self.spec = getattr(legacy_env, "spec", None)

    @staticmethod
    def _convert_space(space):
        if isinstance(space, gym.spaces.Space):
            return space
        if isinstance(space, legacy_gym.spaces.Box):
            return gym.spaces.Box(
                low=np.array(space.low, copy=True),
                high=np.array(space.high, copy=True),
                shape=space.shape,
                dtype=space.dtype,
            )
        if isinstance(space, legacy_gym.spaces.Discrete):
            return gym.spaces.Discrete(space.n, start=space.start)
        raise TypeError(f"Unsupported legacy space type: {type(space)}")

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            seed_fn = getattr(self.legacy_env, "seed", None)
            if callable(seed_fn):
                seed_fn(seed)
        obs = self.legacy_env.reset()
        if isinstance(obs, tuple) and len(obs) == 2:
            return obs
        return obs, {}

    def step(self, action):
        result = self.legacy_env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            return result
        obs, reward, done, info = result
        truncated = False
        if isinstance(info, dict) and "TimeLimit.truncated" in info:
            truncated = bool(info.pop("TimeLimit.truncated"))
        return obs, reward, done, truncated, info

    def render(self, *args, **kwargs):
        if hasattr(self.legacy_env, "render"):
            return self.legacy_env.render(*args, **kwargs)
        return None

    def close(self):
        if hasattr(self.legacy_env, "close"):
            return self.legacy_env.close()


def train(args):
    """Train a PPO/RecurrentPPO policy and save checkpoints under rl/checkpoints/legacy_ppo/."""
    algo_cls, policy_class, policy_kwargs = resolve_algorithm(args)
    algo_tag = args.policy_type
    ckpt_dir = LEGACY_RL_ROOT / "checkpoints" / "legacy_ppo" / args.env / f"{algo_tag}_mlp_test"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_env = Quadcopter3DGates(
        num_envs=args.num_envs,
        randomization=RANDOMIZATION,
        gates_ahead=1,
        initialize_uniform=False,
        initialize_at_random_gates=False,
        seed=args.seed,
        low_obs=args.low_obs,
        no_vel=args.no_vel,
        no_ang_vel=args.no_ang_vel,
        param_input=args.param_input,
        param_input_noise=args.param_input_noise,
    )
    env = VecMonitor(train_env)

    # VecNormalize can be useful for longer experiments, but saved statistics
    # must then be loaded again for rendering/evaluation.
    # env = VecNormalize(
    #     env,
    #     norm_obs=True,
    #     norm_reward=True,
    #     clip_obs=10.0,
    #     clip_reward=10.0,
    #     gamma=0.99  # MUST match your PPO gamma
    # )

    if args.cont:
        model_path = Path(args.cont)
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint '{model_path}' not found.")
        model = algo_cls.load(model_path, env=env, device=args.device, custom_objects=None)
    else:
        algo_kwargs: dict[str, Any] = {}
        model = algo_cls(
            policy_class,
            env,
            learning_rate=args.learning_rate,
            n_steps=args.rollout_fragment_length,
            batch_size=args.batch_size,
            ent_coef=args.entropy_coeff,
            clip_range=args.clip_param,
            # clip_range_vf=args.clip_param_vf,
            vf_coef=args.vf_coeff,
            # max_grad_norm=10,
            gamma=args.gamma,
            gae_lambda=args.lam,
            tensorboard_log=args.tensorboard_log,
            n_epochs=args.n_epochs,
            verbose=1,
            device=args.device,
            policy_kwargs=policy_kwargs,
            **algo_kwargs,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(ckpt_dir),
        name_prefix=f"{algo_tag}_mlp_test",
    )
    attach_gradient_logger(model.policy)
    grad_cb = GradientEpisodePrintCallback()

    start = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, grad_cb], #, pre_backprop_avg_cb],
        progress_bar=True,
    )
    model.save(ckpt_dir / f"{algo_tag}_mlp_test")
    if hasattr(env, "save"):
        env.save(ckpt_dir / f"{algo_tag}_mlp_test.pkl")
    env.close()
    elapsed = (time.time() - start) / 3600
    print(f"Finished training {args.total_timesteps:,} steps in {elapsed:0.2f}h. Latest checkpoint saved to {ckpt_dir}.")


def animate_policy(model, env, reset_func=None, **kwargs):
    """Open the interactive quadcopter animation for a trained policy."""
    obs = env.reset()
    state: tuple[np.ndarray, ...] | None = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    def run():
        nonlocal obs, state, episode_starts

        actions, state = model.predict(
            obs,
            state=state,
            episode_start=episode_starts,
            deterministic=True,
        )
        obs, rewards, dones, infos = env.step(actions)
        episode_starts = dones
        if dones.any():
            state = None
        out = env.render()

        return out
    animation.view(run, gate_pos=env.gate_pos, gate_yaw=env.gate_yaw, fps=1/env.dt, **kwargs)


def render_policy(args):
    """Run a trained checkpoint for several episodes and print evaluation metrics."""
    if not args.cont:
        raise ValueError("--cont must point to a trained checkpoint when using --render.")
    # env = VecMonitor(env)
    algo_cls, policy_class, _ = resolve_algorithm(args)
    test_env = Quadcopter3DGates(
        num_envs=1,
        randomization=RANDOMIZATION,
        gates_ahead=1,
        initialize_uniform=False,
        initialize_at_random_gates=True,
        seed=args.seed,
        low_obs=args.low_obs,
        no_vel=args.no_vel,
        no_ang_vel=args.no_ang_vel,
        param_input=args.param_input,
        param_input_noise=args.param_input_noise,
    )
    env = VecMonitor(test_env)
    model = algo_cls.load(
        args.cont,
        env=env,
        device=args.device,
        custom_objects={"policy_class": policy_class},
    )

    def append_if_present(values: list[float], value: Any) -> None:
        if value is not None:
            values.append(float(value))

    def mean_or_nan(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    animate_policy(model, env) # ACTIVATES AND DISACTIVATES ANIMATION
    obs = env.reset()
    state: tuple[np.ndarray, ...] | None = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    episode_rewards = np.zeros(env.num_envs, dtype=np.float32)
    episode_gates = np.zeros(env.num_envs, dtype=np.int32)
    episode_counts = np.zeros(env.num_envs, dtype=np.int64)
    target_episodes = 200
    total_episodes = 0
    total_reward = 0.0
    total_gates = 0
    total_crashes = 0
    total_gate_crashes = 0
    total_time_limits = 0
    saliency_sums: np.ndarray | None = None
    saliency_steps = np.zeros(env.num_envs, dtype=np.int64)
    saliency_top_k = 10
    time_sim_list = []
    distance_list = []
    max_speed_list = []
    avg_speed_list = []
    avg_bank_angle_list = []
    action_std_list = []
    avg_action_rpm_diff_list = []
    inference_time_list = []
    action_command_sum = 0.0
    action_command_count = 0
    while True:
        # Saliency is accumulated per environment until that environment's
        # episode ends, then the most influential observation indices are shown.
        if args.policy_type in {"recurrent_ppo", "recurrent_ppo_ltc", "recurrent_ppo_ncp_cfc"}:
            step_saliency = _compute_action_saliency_recurrent(
                model.policy,
                obs,
                state,
                episode_starts,
            )
        else:
            step_saliency = _compute_action_saliency(model.policy, obs)
        if saliency_sums is None:
            saliency_sums = np.zeros_like(step_saliency, dtype=np.float64)
        saliency_sums += step_saliency
        saliency_steps += 1

        if args.policy_type in {"recurrent_ppo", "recurrent_ppo_ltc", "recurrent_ppo_ncp_cfc"}:
            predict_start = time.perf_counter()
            action, state = model.predict(
                obs,
                state=state,
                deterministic=True,
            )
        else:
            predict_start = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
        if th.cuda.is_available() and str(model.device).startswith("cuda"):
            th.cuda.synchronize(model.device)
        inference_time = time.perf_counter() - predict_start
        inference_time_list.append(inference_time / env.num_envs)
        action_array = np.asarray(action, dtype=np.float64)
        action_command_sum += float(np.sum(action_array))
        action_command_count += int(action_array.size)

        obs, rewards, dones, infos = env.step(action)
        episode_rewards += rewards
        for i, info in enumerate(infos):
            if info.get("gate_passed", False):
                episode_gates[i] += 1

        if np.any(dones):
            for i, done in enumerate(dones):
                if not done:
                    continue
                info = infos[i]
                crashed = bool(
                    info.get("ground_collision", False)
                    or info.get("out_of_bounds", False)
                    or info.get("gate_collision", False)
                )
                append_if_present(time_sim_list, info.get("TimeLimit.time"))
                append_if_present(distance_list, info.get("distance_travelled"))
                append_if_present(max_speed_list, info.get("max_speed"))
                append_if_present(avg_speed_list, info.get("avg_speed"))
                append_if_present(avg_bank_angle_list, info.get("avg_bank_angle"))
                append_if_present(action_std_list, info.get("action_std"))
                append_if_present(avg_action_rpm_diff_list, info.get("avg_action_rpm_diff"))
                gate_crashed = bool(info.get("gate_collision", False))
                if crashed:
                    if gate_crashed:
                        outcome = "gate-crashed"
                    else:
                        outcome = "out-of-bounds crashed"
                elif info.get("TimeLimit.truncated", False):
                    outcome = "finished (time limit)"
                else:
                    outcome = "finished"
                if saliency_sums is not None and saliency_steps[i] > 0:
                    avg_saliency = saliency_sums[i] / saliency_steps[i]
                    top_idx = np.argsort(-avg_saliency)[:saliency_top_k]
                    top_pairs = ", ".join(f"{idx}:{avg_saliency[idx]:.3e}" for idx in top_idx)
                    print(f"Saliency top{saliency_top_k} (|d logp / d obs|): {top_pairs}")
                    saliency_sums[i].fill(0.0)
                    saliency_steps[i] = 0
                print(
                    f"Episode {episode_counts[i] + 1} | "
                    f"reward={episode_rewards[i]:.3f} | "
                    f"{outcome} | gates_passed={episode_gates[i]}"
                )
                episode_counts[i] += 1
                total_episodes += 1
                total_reward += float(episode_rewards[i])
                total_gates += int(episode_gates[i])
                if crashed:
                    total_crashes += 1
                if gate_crashed:
                    total_gate_crashes += 1
                if info.get("TimeLimit.truncated", False):
                    total_time_limits += 1
                episode_rewards[i] = 0.0
                episode_gates[i] = 0

                if total_episodes >= target_episodes:
                    avg_reward = total_reward / total_episodes
                    avg_gates = total_gates / total_episodes
                    crash_rate = total_crashes / total_episodes
                    gate_crash_rate = total_gate_crashes / total_episodes
                    time_limit_rate = total_time_limits / total_episodes
                    avg_time_sim = mean_or_nan(time_sim_list)
                    avg_distance = mean_or_nan(distance_list)
                    avg_max_speed = mean_or_nan(max_speed_list)
                    avg_speed = mean_or_nan(avg_speed_list)
                    avg_bank_angle = mean_or_nan(avg_bank_angle_list)
                    avg_action_command = (
                        action_command_sum / action_command_count
                        if action_command_count > 0
                        else float("nan")
                    )
                    avg_action_std = mean_or_nan(action_std_list)
                    avg_action_rpm_diff = mean_or_nan(avg_action_rpm_diff_list)
                    avg_inference_time_ms = 1000.0 * mean_or_nan(inference_time_list)
                    finish_rate = 1.0 - crash_rate
                    print(
                        f"\nAverages over {total_episodes} episodes | "
                        f"avg_reward={avg_reward:.3f} | "
                        f"avg_gates_passed={avg_gates:.3f} | "
                        f"crash_rate={crash_rate:.3f} | "
                        f"gate_crash_rate={gate_crash_rate:.3f} | "
                        f"finish_rate={finish_rate:.3f} | "
                        f"time_limit_rate={time_limit_rate:.3f} | "
                        f"avg_time_sim={avg_time_sim:.3f} | "
                        f"avg_distance={avg_distance:.3f} | "
                        f"avg_max_speed={avg_max_speed:.3f} | "
                        f"avg_speed={avg_speed:.3f} | "
                        f"avg_bank_angle={avg_bank_angle:.3f} | "
                        f"avg_action_command={avg_action_command:.3f} | "
                        f"avg_action_std={avg_action_std:.3f} | "
                        f"avg_action_rpm_diff={avg_action_rpm_diff:.3f} | "
                        f"avg_inference_time_ms={avg_inference_time_ms:.3f}\n"
                    )
                    return
        if args.policy_type in {"recurrent_ppo", "recurrent_ppo_ltc", "recurrent_ppo_ncp_cfc"}:
            episode_starts = dones
            if dones.any():
                state = None


def parse_args():
    """Parse command-line options for training or checkpoint evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Quadcopter3DGatesGym-v0")
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-log-std", type=float, default=1000.0)
    parser.add_argument("--rollout-fragment-length", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    # parser.add_argument("--clip-param-vf", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--vf-coeff", type=float, default=0.5)
    parser.add_argument("--total-timesteps", type=int, default=500_000_000)
    parser.add_argument("--checkpoint-freq", type=int, default=500_000)
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default=str(LEGACY_RL_OUTPUT_ROOT / "tensorboard" / "legacy_ppo"),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--cfc-timespan", type=float, default=0.01)
    parser.add_argument(
        "--use-flatten-features",
        type=bool, default=True,
        help="When set, skip the custom CfC feature extractor for recurrent PPO and use the default Flatten extractor.",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        choices=("ppo", "recurrent_ppo", "recurrent_ppo_ltc", "recurrent_ppo_ncp_cfc"),
        default="recurrent_ppo",
        help="Selects between standard PPO (feedforward), Recurrent PPO with CfC, and Recurrent PPO with LTC.",
    )
    parser.add_argument("--ncp-inter-neurons", type=int, default=32)
    parser.add_argument("--ncp-command-neurons", type=int, default=24)
    parser.add_argument("--ncp-sensory-fanout", type=int, default=20)
    parser.add_argument("--ncp-inter-fanout", type=int, default=16)
    parser.add_argument("--ncp-recurrent-command-synapses", type=int, default=16)
    parser.add_argument("--ncp-motor-fanin", type=int, default=20)
    parser.add_argument("--ncp-scale-factor", type=float, default=1.0)
    parser.add_argument(
        "--use-burn-in",
        action="store_true",
        help="Use RecurrentPPOWithBurnIn (burn-in masking) when training recurrent PPO.",
    )
    parser.add_argument(
        "--burn-in-steps",
        type=int,
        default=16,
        help="Number of burn-in steps to ignore in the loss when --use-burn-in is set.",
    )
    parser.add_argument("--cont", type=str, default="")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--low-obs", action="store_true")
    parser.add_argument("--no-vel", action="store_true")
    parser.add_argument("--no-ang-vel", action="store_true")
    parser.add_argument("--param-input", action="store_true")
    parser.add_argument("--param-input-noise", type=float, default=0.0)
    args = parser.parse_args()
    if args.policy_type not in {"recurrent_ppo", "recurrent_ppo_ltc", "recurrent_ppo_ncp_cfc"} and args.use_burn_in:
        raise ValueError("--use-burn-in is only supported with recurrent policy types.")
    return args


# Main entry point
if __name__ == "__main__":
    arguments = parse_args()
    if arguments.render:
        render_policy(arguments)
    else:
        train(arguments)
