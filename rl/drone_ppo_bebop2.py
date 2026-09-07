#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train PPO/SB3 on the Bebop2 waypoint environment.

This is the non-legacy training entrypoint. It uses:

- Bebop2WaypointEnv, not Quadcopter3DGates
- 19 observations, matching the supervised-learning input interface
- 4 actions in [0, 1], matching the supervised-learning output convention
- quadrotor_sim_matlab dynamics through the environment
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

_RL_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lnn_rl_matplotlib")

import numpy as np
import torch as th
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO

from .bc_policy import BCInitializedActorCriticPolicy
from .envs.bebop2_figure8_gates_env import Bebop2Figure8GatesEnv
from .envs.bebop2_waypoints_env import DEFAULT_SQUARE_WAYPOINTS, Bebop2WaypointEnv, ResidualBebop2WaypointEnv
from .legacy_ppo.quadcopter_animation import animation as legacy_animation
from .legacy_ppo.drone_ppo_sb3 import (
    GradientEpisodePrintCallback,
    RecurrentActorCriticCfCPolicy,
    attach_gradient_logger,
    resolve_algorithm,
)
from ..utils.animation import animate
from ..utils.dynamics_models import quadrotor_sim_matlab, quadrotor_sim_matlab_randomized
from ..utils.networks_LNN import CFC, ConvCfC
from ..utils.quadrotor_sim import body_to_world_state


RL_ROOT = _RL_ROOT
PROJECT_ROOT = RL_ROOT.parent
RL_OUTPUT_ROOT = PROJECT_ROOT / "organized_plots" / "rl_runs"


@contextmanager
def _torch_load_zipfile_compat():
    """Make nested PyTorch checkpoints readable from an SB3 ZIP archive.

    SB3 passes a ``zipfile.ZipExtFile`` to ``torch.load``.  Some PyTorch
    versions can inspect the nested ZIP but fail while reading one of its
    records.  Materializing that member as a regular temporary file avoids
    the incompatible nested-stream path without modifying the checkpoint.
    """
    import torch as th

    original_load = th.load

    @wraps(original_load)
    def load_from_regular_file_if_needed(file_or_path, *args, **kwargs):
        if isinstance(file_or_path, (str, os.PathLike)):
            return original_load(file_or_path, *args, **kwargs)

        read = getattr(file_or_path, "read", None)
        if read is None:
            return original_load(file_or_path, *args, **kwargs)

        payload = read()
        with tempfile.NamedTemporaryFile(suffix=".pth") as temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            return original_load(temporary_file.name, *args, **kwargs)

    th.load = load_from_regular_file_if_needed
    try:
        yield
    finally:
        th.load = original_load


def _track_artifact_name(track: str) -> str:
    return "bebop2_waypoints" if track == "square_waypoints" else track


def _canonical_policy_type(policy_type: str) -> str:
    return {"mlp": "ppo", "cfc": "recurrent_ppo"}.get(
        policy_type, policy_type
    )


class _ContinuousTimeRNN(th.nn.Module):
    """Euler-discretized leaky continuous-time recurrent neural network."""

    def __init__(self, input_size: int, hidden_size: int, time_constant: float):
        super().__init__()
        if time_constant <= 0.0:
            raise ValueError("--ctrnn-time-constant must be positive.")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.state_size = self.hidden_size
        self.output_size = self.hidden_size
        self.num_layers = 1
        self.time_constant = float(time_constant)
        self.cfc_timespan = 0.01
        self.input_linear = th.nn.Linear(self.input_size, self.hidden_size)
        self.recurrent_linear = th.nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, inputs: th.Tensor, hidden: th.Tensor):
        if hidden.ndim == 3:
            hidden = hidden.squeeze(0)
        dt = float(self.cfc_timespan)
        alpha = dt / self.time_constant
        if not 0.0 < alpha <= 1.0:
            raise ValueError(
                f"CT-RNN requires 0 < dt <= time constant; got dt={dt:g}, "
                f"time_constant={self.time_constant:g}."
            )
        outputs = []
        for step_inputs in inputs.unbind(dim=1):
            equilibrium = th.tanh(self.input_linear(step_inputs) + self.recurrent_linear(hidden))
            hidden = hidden + alpha * (equilibrium - hidden)
            outputs.append(hidden)
        return th.stack(outputs, dim=1), hidden


class _SimpleRecurrentPolicy(RecurrentActorCriticCfCPolicy):
    """Compatibility layer for one-state recurrent cells in Recurrent-PPO."""

    @staticmethod
    def _process_sequence(features, lstm_states, episode_starts, rnn_module):
        hidden, cell = lstm_states
        if hidden.ndim == 3:
            hidden = hidden.squeeze(0)
        n_seq = hidden.shape[0]
        seq_features = features.reshape((n_seq, -1, rnn_module.input_size)).swapaxes(0, 1)
        seq_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        outputs = []
        for step_features, step_start in zip(seq_features, seq_starts, strict=True):
            hidden = (1.0 - step_start).unsqueeze(-1) * hidden
            step_output, hidden_out = rnn_module(step_features.unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden_out.squeeze(0) if hidden_out.ndim == 3 else hidden_out
            outputs.append(step_output[:, -1, :])
        flattened = th.flatten(th.stack(outputs).swapaxes(0, 1), start_dim=0, end_dim=1)
        return flattened, (hidden.unsqueeze(0), cell)


class RecurrentActorCriticRNNPolicy(_SimpleRecurrentPolicy):
    """Recurrent-PPO actor using a vanilla tanh RNN."""

    def _make_recurrent_cell(self, input_size: int):
        return th.nn.RNN(input_size, self.lstm_output_dim, nonlinearity="tanh", batch_first=True)


class RecurrentActorCriticCTRNNPolicy(_SimpleRecurrentPolicy):
    """Recurrent-PPO actor using a leaky continuous-time RNN."""

    def __init__(self, *args, ctrnn_time_constant: float = 0.10, **kwargs):
        self._ctrnn_time_constant = float(ctrnn_time_constant)
        super().__init__(*args, **kwargs)

    def _make_recurrent_cell(self, input_size: int):
        return _ContinuousTimeRNN(input_size, self.lstm_output_dim, self._ctrnn_time_constant)


class _RLConvCfC(ConvCfC):
    """Expose the project ConvCfC through the interface expected by Recurrent-PPO."""

    def __init__(self, observation_size: int, hidden_size: int, **cfc_kwargs):
        recurrent_core = CFC(
            256,
            hidden_size,
            batch_first=True,
            return_sequences=True,
            **cfc_kwargs,
        )
        super().__init__(
            no_input=1,
            rnn_module=recurrent_core,
            width="base",
            base_channels=256,
        )
        self.input_size = int(observation_size)
        self.state_size = int(recurrent_core.state_size)
        self.output_size = int(recurrent_core.output_size)
        self.hidden_size = self.state_size
        self.num_layers = 1
        self.cfc_timespan = None

    def named_parameters(self, *args, **kwargs):
        # The shared policy initializer applies orthogonal initialization to
        # parameters named "weight". BatchNorm scales are one-dimensional and
        # cannot be initialized that way, so expose a neutral name for them.
        for name, parameter in super().named_parameters(*args, **kwargs):
            if "weight" in name and parameter.ndim < 2:
                name = name.replace("weight", "scale")
            yield name, parameter


class RecurrentActorCriticConvCfCPolicy(RecurrentActorCriticCfCPolicy):
    """Recurrent-PPO actor using utils.networks_LNN.ConvCfC."""

    def _make_recurrent_cell(self, input_size: int):
        return _RLConvCfC(
            observation_size=input_size,
            hidden_size=self.lstm_output_dim,
            **self._cfc_kwargs,
        )

    @staticmethod
    def _process_sequence(features, lstm_states, episode_starts, rnn_module):
        hidden, cell = lstm_states
        hidden = hidden.squeeze(0)
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)

        n_seq = hidden.shape[0]
        sequence = features.reshape((n_seq, -1, rnn_module.input_size)).swapaxes(0, 1)
        starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        outputs = []
        for step_features, step_start in zip(sequence, starts, strict=True):
            hidden = (1.0 - step_start).unsqueeze(-1) * hidden
            conv_input = step_features[:, None, None, :]
            timespan = None
            if rnn_module.cfc_timespan is not None:
                timespan = th.full(
                    (step_features.shape[0], 1, 1),
                    float(rnn_module.cfc_timespan),
                    dtype=step_features.dtype,
                    device=step_features.device,
                )
            step_output, hidden = rnn_module(conv_input, hidden, timespans=timespan)
            outputs.append(step_output[:, -1, :])

        flattened = th.flatten(th.stack(outputs).swapaxes(0, 1), start_dim=0, end_dim=1)
        return flattened, (hidden.unsqueeze(0), cell)


def _state_to_absolute_position(state: np.ndarray, target: np.ndarray) -> np.ndarray:
    return target + body_to_world_state(state)[0:3]


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _action_output_stem(args: argparse.Namespace) -> Path:
    if args.action_plot_output:
        output = Path(args.action_plot_output)
        return output.with_suffix("") if output.suffix else output
    checkpoint_stem = _safe_filename(Path(args.cont).stem)
    track_artifact = _track_artifact_name(args.track)
    return RL_OUTPUT_ROOT / "action_plots" / track_artifact / args.policy_type / checkpoint_stem


def _signals_output_path(args: argparse.Namespace) -> Path:
    if args.signals_plot_output:
        output = Path(args.signals_plot_output)
        return output if output.suffix else output.with_suffix(".png")
    checkpoint_stem = _safe_filename(Path(args.cont).stem)
    track_artifact = _track_artifact_name(args.track)
    return RL_OUTPUT_ROOT / "signal_plots" / track_artifact / args.policy_type / f"{checkpoint_stem}.png"


def _commands01_to_rpm(actions01: np.ndarray) -> np.ndarray:
    info = quadrotor_sim_matlab.INFO
    return info.omega_min + np.clip(actions01, 0.0, 1.0) * (info.omega_max - info.omega_min)


def _synchronize_recurrent_timespan(model, args: argparse.Namespace) -> None:
    """Use the environment dt as the sole time source for loaded CfC/LTC policies."""
    if _canonical_policy_type(args.policy_type) not in {"ct_rnn", "conv_cfc", "recurrent_ppo", "recurrent_ppo_ltc", "recurrent_ppo_ncp_cfc"}:
        return

    if (
        _canonical_policy_type(args.policy_type)
        in {"conv_cfc", "recurrent_ppo", "recurrent_ppo_ltc", "recurrent_ppo_ncp_cfc"}
        and not args.cfc_use_dt
    ):
        return

    timespan = float(args.dt)
    if timespan <= 0.0:
        raise ValueError("--dt must be positive.")

    policy = model.policy
    updated_modules: list[str] = []
    for module_name in ("lstm_actor", "lstm_critic"):
        recurrent_module = getattr(policy, module_name, None)
        if recurrent_module is None:
            continue
        recurrent_module.cfc_timespan = timespan
        updated_modules.append(module_name)

    if not updated_modules:
        raise RuntimeError(
            f"Policy type {args.policy_type!r} has no continuous-time recurrent module to update."
        )

    # Keep both the live policy and future checkpoints consistent with --dt.
    policy._cfc_timespan = timespan
    if isinstance(getattr(model, "policy_kwargs", None), dict):
        model.policy_kwargs["cfc_timespan"] = timespan
    print(
        f"Unified recurrent timestep: dt=timespan={timespan:.9g}s "
        f"({1.0 / timespan:.3f} Hz) | modules={','.join(updated_modules)}"
    )


def _print_action_stats(label: str, actions01: np.ndarray, rpm: np.ndarray) -> None:
    if len(actions01) < 2:
        print(f"{label}: not enough action samples for delta stats.")
        return
    delta_cmd = np.diff(actions01, axis=0)
    delta_rpm = np.diff(rpm, axis=0)
    max_abs_delta_cmd = np.max(np.abs(delta_cmd), axis=0)
    mean_abs_delta_cmd = np.mean(np.abs(delta_cmd), axis=0)
    max_abs_delta_rpm = np.max(np.abs(delta_rpm), axis=0)
    mean_abs_delta_rpm = np.mean(np.abs(delta_rpm), axis=0)
    stats = []
    for idx in range(4):
        stats.append(
            f"u{idx + 1}: max_delta={max_abs_delta_cmd[idx]:.4f} "
            f"({max_abs_delta_rpm[idx]:.1f} rpm), "
            f"mean_delta={mean_abs_delta_cmd[idx]:.4f} ({mean_abs_delta_rpm[idx]:.1f} rpm)"
        )
    print(f"{label} action smoothness | " + " | ".join(stats))


def _save_action_csv(path: Path, rollouts: list[dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "episode",
                "step",
                "time_s",
                "u1_rpm",
                "u2_rpm",
                "u3_rpm",
                "u4_rpm",
                "du1_rpm",
                "du2_rpm",
                "du3_rpm",
                "du4_rpm",
            ]
        )
        for episode_idx, rollout in enumerate(rollouts, start=1):
            actions01 = np.asarray(rollout["actions"], dtype=np.float64)
            rpm = _commands01_to_rpm(actions01)
            delta_rpm = np.vstack([np.zeros((1, 4), dtype=np.float64), np.diff(rpm, axis=0)])
            for step_idx, (time_s, action_row, rpm_row, delta_row) in enumerate(
                zip(rollout["t"], actions01, rpm, delta_rpm, strict=True)
            ):
                writer.writerow([episode_idx, step_idx, time_s, *rpm_row, *delta_row])


def _plot_action_rollouts(
    rollouts: list[dict[str, np.ndarray]],
    args: argparse.Namespace,
    title: str,
) -> None:
    if not rollouts:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_stem = _action_output_stem(args)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    csv_path = output_stem.with_suffix(".csv")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for episode_idx, rollout in enumerate(rollouts, start=1):
        actions01 = np.asarray(rollout["actions"], dtype=np.float64)
        rpm = _commands01_to_rpm(actions01)
        time_s = np.asarray(rollout["t"], dtype=np.float64)
        alpha = 0.95 if len(rollouts) == 1 else 0.35
        for motor_idx in range(4):
            suffix = "" if len(rollouts) == 1 else f" ep{episode_idx}"
            axes[motor_idx].plot(
                time_s,
                rpm[:, motor_idx],
                color=colors[motor_idx],
                alpha=alpha,
                linewidth=1.1,
                label=f"motor {motor_idx + 1}{suffix}",
            )
        _print_action_stats(f"Episode {episode_idx}", actions01, rpm)

    for motor_idx, ax in enumerate(axes):
        ax.set_ylabel(f"motor {motor_idx + 1}\n[rpm]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    _save_action_csv(csv_path, rollouts)
    print(f"Action plot saved to {png_path}")
    print(f"Action CSV saved to {csv_path}")


def make_env(
    args: argparse.Namespace,
    num_envs: int,
    seed: int | None = None,
    terminate_on_waypoint: bool | None = None,
):
    if args.track == "figure8_gates":
        if args.policy_type in {"bc_ppo", "residual_ppo"}:
            raise ValueError(
                "--track figure8_gates uses legacy-style observations; "
                "choose --policy-type ppo, recurrent_ppo, recurrent_ppo_ltc, or recurrent_ppo_ncp_cfc."
            )
        figure8_start_pos = None
        figure8_start_pos_jitter = 0.5
        figure8_start_pos_enu = getattr(args, "figure8_start_pos_enu", None)
        if figure8_start_pos_enu is not None:
            x_enu, y_enu, z_enu = figure8_start_pos_enu
            # Paparazzi/Gazebo ENU -> controller world frame {Y, X, -Z}.
            figure8_start_pos = np.asarray([y_enu, x_enu, -z_enu], dtype=np.float32)
            figure8_start_pos_jitter = 0.0

        env_kwargs = dict(
            num_envs=num_envs,
            start_gate=getattr(args, "figure8_start_gate", 0),
            gates_ahead=args.gates_ahead,
            gate_size=args.gate_size,
            dt=args.dt,
            tau=args.tau,
            max_steps=args.max_steps,
            integration_method=args.integration_method,
            implicit_iters=args.implicit_iters,
            initialize_at_random_gates=args.initialize_at_random_gates,
            initialize_uniform=args.initialize_uniform,
            num_state_history=args.num_state_history,
            num_action_history=args.num_action_history,
            history_step_size=args.history_step_size,
            param_input=args.param_input,
            param_input_noise=args.param_input_noise,
            motor_tau=args.motor_tau,
            rotor_yaw_sign=args.rotor_yaw_sign,
            obs_rate_noise_std=args.obs_rate_noise_std,
            invert_yaw_observation=args.invert_yaw_observation,
            low_obs=args.low_obs,
            no_vel=args.no_vel,
            no_ang_vel=args.no_ang_vel,
            randomize_external_moments=args.randomize_external_moments,
            action_range=args.figure8_action_range,
            randomize_dynamics=args.randomize_dynamics,
            randomization_factor=args.randomization_factor,
            randomize_aerodynamic_coefficients=args.randomize_aerodynamic_coefficients,
            seed=seed,
        )
        if figure8_start_pos is not None:
            env_kwargs["start_pos"] = figure8_start_pos
            env_kwargs["start_pos_jitter"] = figure8_start_pos_jitter
        return Bebop2Figure8GatesEnv(**env_kwargs)
    if terminate_on_waypoint is None:
        terminate_on_waypoint = True
    normalize_observations = bool(args.normalize_observations and _canonical_policy_type(args.policy_type) in {"ppo", "recurrent_ppo", "recurrent_ppo_ltc", "recurrent_ppo_ncp_cfc"})
    env_kwargs = dict(
        num_envs=num_envs,
        waypoint_radius=args.dist_error,
        dt=args.dt,
        max_steps=args.max_steps,
        integration_method=args.integration_method,
        implicit_iters=args.implicit_iters,
        initialize_at_random_waypoints=args.initialize_at_random_waypoints,
        terminate_on_waypoint=terminate_on_waypoint,
        normalize_observations=normalize_observations,
        randomize_external_moments=args.randomize_external_moments,
        point_to_point=args.track == "point_to_point",
        success_speed=args.success_speed,
        success_attitude=args.success_attitude,
        motor_tau=args.tau,
        seed=seed,
    )
    if args.policy_type == "residual_ppo":
        return ResidualBebop2WaypointEnv(
            **env_kwargs,
            bc_config_path=args.bc_config,
            bc_checkpoint_path=args.bc_checkpoint,
            project_root=PROJECT_ROOT,
            residual_scale=args.residual_scale,
            device=args.device if args.device != "auto" else "cpu",
        )
    return Bebop2WaypointEnv(**env_kwargs)


def train(args: argparse.Namespace) -> None:
    algo_cls, policy_class, policy_kwargs = resolve_bebop2_algorithm(args)
    algo_tag = args.policy_type
    output_tag = args.output_tag if hasattr(args, "output_tag") and args.output_tag else "ckpt"
    track_artifact = _track_artifact_name(args.track)
    ckpt_dir = RL_ROOT / "checkpoints" / track_artifact / algo_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.experiment_id or f"{track_artifact}_{algo_tag}_seed{args.seed}_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir = RL_ROOT / "runs" / track_artifact / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    metadata = vars(args).copy()
    final_checkpoint = ckpt_dir / f"{run_id}.zip"
    figure8 = args.track == "figure8_gates"
    metadata.update(
        experiment_id=run_id,
        git_commit=git_commit,
        model=algo_tag,
        training_method="PPO",
        dynamics_model="corrected_bebop2_matlab",
        actuator_tau=args.motor_tau if figure8 else args.tau,
        action_mapping=(
            f"four motor commands; figure8 policy range={args.figure8_action_range}"
            if figure8
            else "four motor commands [0,1] -> [omega_min,omega_max]"
        ),
        observation_configuration=(
            f"figure8 gate-relative observation; no_vel={args.no_vel}; "
            f"no_ang_vel={args.no_ang_vel}; low_obs={args.low_obs}"
            if figure8
            else "19 target-relative states, BEBOP2_BC_BASELINE normalization"
        ),
        checkpoint=str(final_checkpoint),
        final_evaluation_results=None,
    )
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    train_env = make_env(args, num_envs=args.num_envs, seed=args.seed)
    env = VecMonitor(train_env)

    if args.cont:
        model_path = Path(args.cont)
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint '{model_path}' not found.")
        with _torch_load_zipfile_compat():
            model = algo_cls.load(
                model_path,
                env=env,
                device=args.device,
                custom_objects={"policy_class": policy_class},
            )
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
            vf_coef=args.vf_coeff,
            gamma=args.gamma,
            gae_lambda=args.lam,
            tensorboard_log=str(run_dir),
            n_epochs=args.n_epochs,
            verbose=1,
            device=args.device,
            policy_kwargs=policy_kwargs,
            **algo_kwargs,
        )

    _synchronize_recurrent_timespan(model, args)

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(ckpt_dir),
        name_prefix=run_id,
    )
    attach_gradient_logger(model.policy)
    grad_cb = GradientEpisodePrintCallback()

    start = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, grad_cb],
        progress_bar=True,
    )
    model.save(final_checkpoint)
    env.close()
    elapsed = (time.time() - start) / 3600
    print(f"Finished training {args.total_timesteps:,} steps in {elapsed:0.2f}h. Latest checkpoint saved to {ckpt_dir}.")


def validate_environment(args: argparse.Namespace) -> None:
    """Exercise reset, hover/random actions, rewards, and terminal flags without learning."""
    env = make_env(args, num_envs=4, seed=args.seed)
    obs = env.reset()
    assert obs.shape == (4, env.observation_space.shape[0]) and np.isfinite(obs).all()
    if args.track == "figure8_gates":
        for actions in (
            np.full((4, 4), 0.5, dtype=np.float32),
            env._rng.uniform(env.action_space.low, env.action_space.high, size=(4, 4)).astype(np.float32),
        ):
            env.step_async(actions)
            next_obs, rewards, _, infos = env.step_wait()
            assert np.isfinite(next_obs).all() and np.isfinite(rewards).all()
            assert all("gate_passed" in info and "ground_collision" in info for info in infos)
        print(
            f"Environment validation passed: track={args.track}, obs={env.observation_space.shape}, "
            f"action={env.action_space.shape}, dt={args.dt:g}s, tau={args.motor_tau:g}s; no training launched."
        )
        env.close()
        return

    hover = quadrotor_sim_matlab.INFO.u_hover
    env.states[0] = 0.0
    env.states[0, 15:19] = quadrotor_sim_matlab.INFO.hover_omega
    env.states[1, 6] = np.pi / 2.0 + 0.1
    env.step_async(np.full((4, 4), hover, dtype=np.float32))
    next_obs, rewards, dones, infos = env.step_wait()
    assert np.isfinite(next_obs).all() and np.isfinite(rewards).all()
    assert dones[0] and infos[0]["is_success"], "stable target state was not successful"
    assert dones[1] and infos[1]["attitude_crash"], "unsafe attitude did not terminate"
    env.step_async(env.rng.uniform(0.0, 1.0, size=(4, 4)).astype(np.float32))
    next_obs, rewards, _, infos = env.step_wait()
    assert np.isfinite(next_obs).all() and np.isfinite(rewards).all()
    assert all("is_success" in info and "ground_collision" in info for info in infos)
    print(
        f"Environment validation passed: track={args.track}, obs={env.observation_space.shape}, "
        f"action={env.action_space.shape} in [0,1], dt={args.dt:g}s, tau={args.tau:g}s; no training launched."
    )
    env.close()


def _collect_render_episode(model, args: argparse.Namespace):
    env = make_env(args, num_envs=1, seed=args.seed, terminate_on_waypoint=False)
    obs = env.reset()
    state = None
    episode_start = np.ones((env.num_envs,), dtype=bool)
    positions = []
    orientations = []
    actions_history = []
    targets = []
    rewards = []

    for _ in range(args.render_steps):
        action, state = model.predict(
            obs,
            state=state,
            episode_start=episode_start,
            deterministic=True,
        )
        obs, reward, done, infos = env.step(action)
        target = env._target_positions()[0].astype(np.float64)
        positions.append(_state_to_absolute_position(env.states[0], target))
        orientations.append(env.states[0, 6:9].copy())
        actions_history.append(env.actions[0].astype(np.float64))
        targets.append(target.copy())
        rewards.append(float(reward[0]))
        reached_waypoint = bool(infos[0].get("waypoint_reached", False))
        episode_start = done | (args.reset_recurrent_at_waypoint and np.asarray([reached_waypoint], dtype=bool))
        if bool(done[0]):
            break

    if not positions:
        raise RuntimeError("No rollout samples were collected for rendering.")

    rollout = {
        "positions": np.asarray(positions, dtype=np.float64),
        "orientations": np.asarray(orientations, dtype=np.float64),
        "actions": np.asarray(actions_history, dtype=np.float64),
        "targets": np.asarray(targets, dtype=np.float64),
        "t": np.arange(len(positions), dtype=np.float64) * args.dt,
        "reward": float(sum(rewards)),
    }
    env.close()
    return rollout


def render(args: argparse.Namespace) -> None:
    if not args.cont:
        raise ValueError("--render requires --cont pointing to a trained SB3 checkpoint.")

    algo_cls, policy_class, _ = resolve_bebop2_algorithm(args)
    if args.track == "figure8_gates":
        env = make_env(args, num_envs=1, seed=args.seed)
        with _torch_load_zipfile_compat():
            model = algo_cls.load(
                args.cont,
                env=env,
                device=args.device,
                custom_objects={"policy_class": policy_class},
            )
        _synchronize_recurrent_timespan(model, args)
        obs = env.reset()
        state = None
        episode_start = np.ones((env.num_envs,), dtype=bool)
        figure8_actions = []
        figure8_states_world = []
        figure8_times = []
        figure8_step = 0

        def run():
            nonlocal obs, state, episode_start, figure8_step
            actions, state = model.predict(
                obs,
                state=state,
                episode_start=episode_start,
                deterministic=True,
            )
            obs, _, dones, _ = env.step(actions)
            figure8_actions.append(env.action_to_bebop_command(env.actions[0].astype(np.float64)))
            target = env.gate_pos[env.target_gates[0] % env.num_gates].astype(np.float64)
            state_world = body_to_world_state(env.sim_states[0].astype(np.float64))
            state_world[0:3] += target
            figure8_states_world.append(state_world)
            figure8_times.append(figure8_step * args.dt)
            figure8_step += 1
            episode_start = dones
            if dones.any():
                state = None
            return env.render()

        legacy_animation.view(
            run,
            gate_pos=env.gate_pos,
            gate_yaw=env.gate_yaw,
            fps=1 / env.dt,
            record_steps=args.render_steps if args.record else 0,
            record_file=args.output,
            show_window=args.auto_play or not args.record,
        )
        if args.plot_actions and figure8_actions:
            _plot_action_rollouts(
                [
                    {
                        "actions": np.asarray(figure8_actions, dtype=np.float64),
                        "t": np.asarray(figure8_times, dtype=np.float64),
                    }
                ],
                args,
                title=f"{args.track} {args.policy_type} actions - {Path(args.cont).name}",
            )
        if args.plot_signals and figure8_actions:
            from ..simulators.Simulator_gazebo_square_C import _plot_all_signals

            signals_output = _signals_output_path(args)
            _plot_all_signals(
                states_world=np.asarray(figure8_states_world, dtype=np.float64),
                actions=np.asarray(figure8_actions, dtype=np.float64),
                dt=args.dt,
                output_path=signals_output,
                title=f"{args.track} {args.policy_type} signals - {Path(args.cont).name}",
            )
            print(f"Signals plot saved to {signals_output}")
        env.close()
        return

    load_env = make_env(args, num_envs=1, seed=args.seed)
    with _torch_load_zipfile_compat():
        model = algo_cls.load(
            args.cont,
            env=load_env,
            device=args.device,
            custom_objects={"policy_class": policy_class},
        )
    _synchronize_recurrent_timespan(model, args)
    load_env.close()

    rollouts = []
    for episode_idx in range(args.render_episodes):
        episode_args = argparse.Namespace(**vars(args))
        episode_args.seed = None if args.seed is None else args.seed + episode_idx
        rollout = _collect_render_episode(model, episode_args)
        rollouts.append(rollout)
        print(
            f"Episode {episode_idx + 1}: "
            f"steps={len(rollout['t'])} | total_reward={rollout['reward']:.3f}"
        )

    if args.plot_actions:
        _plot_action_rollouts(
            rollouts,
            args,
            title=f"{args.track} {args.policy_type} actions - {Path(args.cont).name}",
        )

    if args.render_episodes == 1:
        rollout = rollouts[0]
        positions_arr = rollout["positions"]
        orientations_arr = rollout["orientations"]
        actions_arr = rollout["actions"]
        targets_arr = rollout["targets"]
        t = rollout["t"]

        animate(
            t=t,
            x=positions_arr[:, 0],
            y=positions_arr[:, 1],
            z=positions_arr[:, 2],
            phi=orientations_arr[:, 0],
            theta=orientations_arr[:, 1],
            psi=orientations_arr[:, 2],
            u=actions_arr,
            target=targets_arr,
            waypoints=DEFAULT_SQUARE_WAYPOINTS.astype(np.float64),
            file=args.output,
            record=args.record,
            auto_play=args.auto_play or args.record,
            close_on_end=args.record,
            draw_path=True,
        )
        return

    animate(
        t=[rollout["t"] for rollout in rollouts],
        x=[rollout["positions"][:, 0] for rollout in rollouts],
        y=[rollout["positions"][:, 1] for rollout in rollouts],
        z=[rollout["positions"][:, 2] for rollout in rollouts],
        phi=[rollout["orientations"][:, 0] for rollout in rollouts],
        theta=[rollout["orientations"][:, 1] for rollout in rollouts],
        psi=[rollout["orientations"][:, 2] for rollout in rollouts],
        u=[rollout["actions"] for rollout in rollouts],
        target=[],
        waypoints=DEFAULT_SQUARE_WAYPOINTS.astype(np.float64),
        file=args.output,
        multiple_trajectories=True,
        simultaneous=args.simultaneous,
        names=[f"episode_{idx + 1}" for idx in range(len(rollouts))],
        record=args.record,
        auto_play=args.auto_play or args.record,
        close_on_end=args.record,
        draw_path=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-log-std", type=float, default=-1.5)
    parser.add_argument("--rollout-fragment-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.1)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--vf-coeff", type=float, default=0.5)
    parser.add_argument("--total-timesteps", type=int, default=100_000_000)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--tensorboard-log", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--cfc-timespan", type=float, default=0.01)
    parser.add_argument(
        "--cfc-use-dt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pass the environment --dt to CfC/NCP-CfC cells. Use "
            "--no-cfc-use-dt for the discrete-step CfC timing ablation."
        ),
    )
    parser.add_argument("--use-flatten-features", type=bool, default=True)
    parser.add_argument(
        "--policy-type",
        choices=("mlp", "rnn", "ct_rnn", "cfc", "conv_cfc", "ppo", "recurrent_ppo", "recurrent_ppo_ltc", "recurrent_ppo_ncp_cfc", "bc_ppo", "residual_ppo"),
        default="rnn",
    )
    parser.add_argument(
        "--ctrnn-time-constant",
        type=float,
        default=0.10,
        help="CT-RNN neural time constant in seconds; integration uses --dt.",
    )
    parser.add_argument("--ncp-inter-neurons", type=int, default=32)
    parser.add_argument("--ncp-command-neurons", type=int, default=24)
    parser.add_argument("--ncp-sensory-fanout", type=int, default=20)
    parser.add_argument("--ncp-inter-fanout", type=int, default=16)
    parser.add_argument("--ncp-recurrent-command-synapses", type=int, default=16)
    parser.add_argument("--ncp-motor-fanin", type=int, default=20)
    parser.add_argument("--ncp-scale-factor", type=float, default=1.0)
    parser.add_argument(
        "--bc-config",
        type=str,
        default=str(PROJECT_ROOT / "configs/bebop1/bebop1_conv_cfc_h=64_seq=1_epoch=18_val_loss=0.000142.yaml"),
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=str,
        default="bebop1_conv_cfc_h=64_seq=1_epoch=18_val_loss=0.000142.ckpt",
    )
    parser.add_argument("--bc-value-hidden-dim", type=int, default=64)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--cont", type=str, default="")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=0.06)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--track", choices=("point_to_point", "square_waypoints", "figure8_gates"), default="figure8_gates")
    parser.add_argument("--success-speed", type=float, default=0.25)
    parser.add_argument("--success-attitude", type=float, default=0.30)
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--validate-env", action="store_true", help="Run deterministic environment checks and exit without training.")
    parser.add_argument(
        "--figure8-action-range",
        choices=("0_1", "neg1_1"),
        default="0_1",
        help="Policy action range for --track figure8_gates. Use neg1_1 for old checkpoints.",
    )
    parser.add_argument("--dist-error", type=float, default=0.2, help="Waypoint switch distance in meters.")
    parser.add_argument("--gate-size", type=float, default=1.5)
    parser.add_argument("--gates-ahead", type=int, default=1)
    parser.add_argument("--integration-method", default="rk4")
    parser.add_argument("--implicit-iters", type=int, default=1)
    parser.add_argument("--initialize-at-random-waypoints", action="store_true")
    parser.add_argument("--initialize-at-random-gates", action="store_true")
    parser.add_argument("--initialize-uniform", action="store_true")
    parser.add_argument(
        "--figure8-start-pos-enu",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help=(
            "Exact figure-8 start position in Paparazzi/Gazebo ENU coordinates. "
            "For example: --figure8-start-pos-enu 1.9 1.0 1.0."
        ),
    )
    parser.add_argument(
        "--figure8-start-gate",
        type=int,
        choices=range(8),
        default=0,
        metavar="{0..7}",
        help="First target gate in figure-8 mode: 0=RL_F8_1, ..., 7=RL_F8_8.",
    )
    parser.add_argument("--num-state-history", type=int, default=0)
    parser.add_argument("--num-action-history", type=int, default=0)
    parser.add_argument("--history-step-size", type=int, default=1)
    parser.add_argument("--low-obs", action="store_true")
    parser.add_argument("--no-vel", action="store_true")
    parser.add_argument("--no-ang-vel", action="store_true")
    parser.add_argument("--param-input", action="store_true")
    parser.add_argument("--param-input-noise", type=float, default=0.0)
    parser.add_argument(
        "--motor-tau",
        type=float,
        default=0.06,
        help="Motor-response time constant in seconds for the Python Bebop2 dynamics.",
    )
    parser.add_argument(
        "--rotor-yaw-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Sign applied to the Bebop2 rotor reaction torque in yaw.",
    )
    parser.add_argument(
        "--obs-rate-noise-std",
        type=float,
        nargs=3,
        metavar=("P_STD", "Q_STD", "R_STD"),
        default=(0.0, 0.0, 0.0),
        help="Gaussian standard deviations added to the p/q/r policy observations.",
    )
    parser.add_argument(
        "--invert-yaw-observation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Invert the target-relative yaw supplied to the policy.",
    )
    parser.add_argument("--normalize-observations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize-external-moments", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-steps", type=int, default=2000)
    parser.add_argument("--render-episodes", type=int, default=1)
    parser.add_argument("--reset-recurrent-at-waypoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simultaneous", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--auto-play", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-actions", action="store_true")
    parser.add_argument(
        "--action-plot-output",
        default="",
        help="Optional output path for --plot-actions. Defaults to organized_plots/rl_runs/action_plots/<track>/<policy>/<checkpoint>.png/.csv.",
    )
    parser.add_argument("--plot-signals", action="store_true")
    parser.add_argument(
        "--signals-plot-output",
        default="",
        help="Optional PNG path for --plot-signals. Defaults to organized_plots/rl_runs/signal_plots/<track>/<policy>/<checkpoint>.png.",
    )
    parser.add_argument("--log-std-init", type=float, default=-3.0)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--randomize-dynamics", action="store_true")
    parser.add_argument("--randomization-factor", type=float, default=0.30)
    parser.add_argument("--randomize-aerodynamic-coefficients", action="store_true")
    args = parser.parse_args()
    track_artifact = _track_artifact_name(args.track)
    if not args.tensorboard_log:
        args.tensorboard_log = str(RL_OUTPUT_ROOT / "tensorboard" / track_artifact)
    if not args.output:
        args.output = str(RL_OUTPUT_ROOT / "videos" / track_artifact / f"{track_artifact}_rollout.mp4")
    if args.track == "figure8_gates" and args.policy_type in {"bc_ppo", "residual_ppo"}:
        raise ValueError(
            "--track figure8_gates uses legacy-style observations; "
            "choose --policy-type ppo, recurrent_ppo, recurrent_ppo_ltc, or recurrent_ppo_ncp_cfc."
        )
    return args


def resolve_bebop2_algorithm(args):
    canonical_policy = _canonical_policy_type(args.policy_type)
    if canonical_policy == "conv_cfc":
        policy_kwargs = dict(
            features_extractor_class=FlattenExtractor,
            share_features_extractor=True,
            normalize_images=False,
            shared_lstm=False,
            enable_critic_lstm=False,
            lstm_hidden_size=args.cell_size,
            cfc_timespan=args.dt if args.cfc_use_dt else None,
            cfc_kwargs=dict(mixed_memory=False),
            ncp_kwargs=None,
            max_log_std=args.max_log_std,
        )
        return RecurrentPPO, RecurrentActorCriticConvCfCPolicy, policy_kwargs
    if canonical_policy in {"rnn", "ct_rnn"}:
        policy_kwargs = dict(
            features_extractor_class=FlattenExtractor,
            share_features_extractor=True,
            normalize_images=False,
            shared_lstm=False,
            enable_critic_lstm=False,
            lstm_hidden_size=args.cell_size,
            cfc_timespan=args.dt,
            cfc_kwargs={},
            ncp_kwargs=None,
            max_log_std=args.max_log_std,
        )
        policy_class = RecurrentActorCriticRNNPolicy
        if canonical_policy == "ct_rnn":
            policy_class = RecurrentActorCriticCTRNNPolicy
            policy_kwargs["ctrnn_time_constant"] = args.ctrnn_time_constant
        return RecurrentPPO, policy_class, policy_kwargs
    if args.policy_type == "bc_ppo":
        policy_kwargs = dict(
            bc_config_path=args.bc_config,
            bc_checkpoint_path=args.bc_checkpoint,
            project_root=PROJECT_ROOT,
            value_hidden_dim=args.bc_value_hidden_dim,
            max_log_std=args.max_log_std,
            log_std_init=args.log_std_init,
        )
        return PPO, BCInitializedActorCriticPolicy, policy_kwargs
    if args.policy_type == "residual_ppo":
        ppo_args = argparse.Namespace(**vars(args))
        ppo_args.policy_type = "ppo"
        ppo_args.cfc_timespan = ppo_args.dt
        return resolve_algorithm(ppo_args)
    ppo_args = argparse.Namespace(**vars(args))
    ppo_args.policy_type = canonical_policy
    # The environment always needs --dt for dynamics integration. CfC can
    # optionally omit elapsed time for the controlled timing ablation.
    if canonical_policy in {"recurrent_ppo", "recurrent_ppo_ltc", "recurrent_ppo_ncp_cfc"}:
        ppo_args.cfc_timespan = ppo_args.dt if args.cfc_use_dt else None
    else:
        ppo_args.cfc_timespan = ppo_args.dt
    return resolve_algorithm(ppo_args)


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.validate_env:
        validate_environment(parsed_args)
    elif parsed_args.render:
        render(parsed_args)
    else:
        train(parsed_args)
