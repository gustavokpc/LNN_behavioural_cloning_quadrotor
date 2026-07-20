#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Square-waypoint simulation using a PyTorch checkpoint-backed controller."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Iterable

_RUNS_DIR = Path(__file__).resolve().parent / "runs"
os.environ.setdefault("MPLCONFIGDIR", str(_RUNS_DIR / ".matplotlib"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from Simulator_gazebo_square_C import (
    DEFAULT_INITIAL_MOTOR_RPM,
    _animate_square,
    _enu_to_network_world,
    _network_world_to_enu,
    _override_dynamics_tau,
    _plot_all_signals,
    _square_waypoints,
    _wrap_angle,
)
from utils.config import load_yaml
from utils.data import expand_feature_labels, get_norm_vectors
from utils.dynamics_models import available_dynamics_models, get_dynamics_info, set_dynamics_model
from utils.normalization_limits import resolve_normalization_profile
from utils.quadrotor_sim import (
    build_input_vector,
    build_lightning_model,
    body_to_world_trajectory,
    input_labels_without_time,
    integrate_state,
    make_model_observation,
    update_observation_window,
    world_to_body_state,
)


MODEL_PRESETS = {
    "NOVA_VERSAOZE": (
        "checkpoints/bebop2/NOVA_VERSAOZE_BEBP2_conv_cfc_default_n64_bebop2_epoch=19_val_loss=0.000098.ckpt",
        "configs/bebop2/NOVA_VERSAOZE_BEBP2_conv_cfc_default_n64_bebop2_epoch=19_val_loss=0.000098.yaml",
        "bebop2_tau_0_06",
    ),
    "CFC": (
        "checkpoints/bebop1/new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.ckpt",
        "configs/bebop1/new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.yaml",
        "bebop1",
    ),
}


def _resolve_model(
    project_root: Path,
    model: str,
    checkpoint: Path | None,
    model_config: Path | None,
    normalization_limits: str | None,
) -> tuple[Path, Path, str]:
    model_key = model.upper()
    if checkpoint is None or model_config is None or normalization_limits is None:
        if model_key not in MODEL_PRESETS:
            valid = ", ".join(sorted(MODEL_PRESETS))
            raise KeyError(f"Unknown model '{model}'. Choose one of: {valid}, or pass all explicit paths.")
        default_ckpt, default_config, default_norm = MODEL_PRESETS[model_key]
    else:
        default_ckpt = default_config = default_norm = None

    ckpt_path = checkpoint or project_root / default_ckpt
    config_path = model_config or project_root / default_config
    norm_profile = resolve_normalization_profile(normalization_limits or default_norm)
    return ckpt_path, config_path, norm_profile


def _norm_vectors(labels: list[str], normalization_limits: str) -> tuple[np.ndarray, np.ndarray]:
    mins, maxs = get_norm_vectors(labels, normalization_limits)
    return mins.reshape(-1).astype(np.float64), maxs.reshape(-1).astype(np.float64)


def _normalize(values: np.ndarray, labels: list[str], normalization_limits: str) -> np.ndarray:
    mins, maxs = _norm_vectors(labels, normalization_limits)
    return (np.asarray(values, dtype=np.float64) - mins) / (maxs - mins + 1.0e-10)


def _normalized_window(
    state: np.ndarray,
    input_labels: list[str],
    seq_len: int,
    normalization_limits: str,
) -> np.ndarray:
    obs = _normalize(build_input_vector(state, input_labels), input_labels, normalization_limits)
    return np.repeat(obs[np.newaxis, :], repeats=max(1, seq_len), axis=0)


def _update_window(
    window: np.ndarray,
    state: np.ndarray,
    input_labels: list[str],
    seq_len: int,
    normalization_limits: str,
) -> np.ndarray:
    new_obs = _normalize(build_input_vector(state, input_labels), input_labels, normalization_limits)
    if seq_len <= 1:
        return new_obs[np.newaxis, :]
    return np.concatenate([window[-(seq_len - 1):], new_obs[np.newaxis, :]], axis=0)


def rollout_checkpoint_controller(
    model,
    initial_state: np.ndarray,
    input_labels: list[str],
    dt: float,
    horizon_steps: int,
    device: torch.device,
    use_sequencing: bool,
    seq_len: int,
    normalization_limits: str,
    force_timespans: bool = False,
    integration_method: str = "rk4",
    implicit_iters: int = 1,
    stop_fn: Callable[[np.ndarray, int], bool] | None = None,
    input_noise_std: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base_labels = input_labels_without_time(input_labels)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    window = _normalized_window(state, base_labels, seq_len, normalization_limits)
    dt_tensor = None
    if force_timespans or getattr(model, "with_time", False):
        dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device).reshape(1, 1, 1)

    hx = None
    prev_deriv = None
    states = [state.copy()]
    actions = []

    for step_idx in range(int(horizon_steps)):
        obs_window = window
        if input_noise_std:
            if rng is None:
                rng = np.random.default_rng()
            noisy_obs = window[-1].copy()
            expanded_labels = expand_feature_labels(base_labels)
            mins, maxs = _norm_vectors(base_labels, normalization_limits)
            for input_idx, label in enumerate(expanded_labels):
                sigma = float(input_noise_std.get(label, 0.0))
                if sigma > 0.0:
                    raw_value = noisy_obs[input_idx] * (maxs[input_idx] - mins[input_idx] + 1.0e-10) + mins[input_idx]
                    raw_value += rng.normal(0.0, sigma)
                    noisy_obs[input_idx] = (raw_value - mins[input_idx]) / (maxs[input_idx] - mins[input_idx] + 1.0e-10)
            obs_window = window.copy()
            obs_window[-1] = noisy_obs

        obs_tensor = make_model_observation(obs_window, use_sequencing=use_sequencing, device=device)
        with torch.no_grad():
            output = model(obs_tensor, hx=hx, timespans=dt_tensor)
            prediction, hx = output if isinstance(output, tuple) else (output, None)
        action = torch.clamp(prediction, min=0.0, max=1.0).squeeze(0).squeeze(0).cpu().numpy()
        state, prev_deriv = integrate_state(
            integration_method,
            state,
            action,
            dt,
            prev_deriv=prev_deriv,
            implicit_iters=implicit_iters,
        )
        states.append(state.copy())
        actions.append(action)
        if stop_fn is not None and stop_fn(state, step_idx + 1):
            break
        window = _update_window(window, state, base_labels, seq_len, normalization_limits)

    return np.asarray(states), np.asarray(actions)


def simulate_gazebo_square_python(
    config_model: dict,
    checkpoint_path: Path,
    project_root: Path,
    dt: float,
    time_simulation: float,
    dist_error: float,
    integration_method: str,
    implicit_iters: int,
    start_waypoint_index: int,
    reset_each_waypoint: bool,
    start_alt_m: float,
    waypoint_alt_m: float,
    dynamics_model: str,
    normalization_limits: str,
    tau: float | None = None,
    device_name: str = "cpu",
    input_noise_std: dict[str, float] | None = None,
    input_noise_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    set_dynamics_model(dynamics_model)
    _override_dynamics_tau(tau)
    dynamics_info = get_dynamics_info()
    input_noise_rng = np.random.default_rng(input_noise_seed) if input_noise_std else None
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    if device_name == "auto" and device.type != "cuda":
        device = torch.device("cpu")
    model = build_lightning_model(config_model, str(checkpoint_path), project_root, device)

    standby_enu, square_enu = _square_waypoints(start_alt_m=start_alt_m, waypoint_alt_m=waypoint_alt_m)
    square_world = np.asarray([_enu_to_network_world(wp) for wp in square_enu], dtype=np.float64)
    current_world = np.zeros(19, dtype=np.float64)
    current_world[0:3] = _enu_to_network_world(standby_enu)
    initial_motor_rpm = DEFAULT_INITIAL_MOTOR_RPM if dynamics_info.name == "quadrotor_sim_matlab" else dynamics_info.omega_mid
    current_world[15:19] = initial_motor_rpm

    max_total_steps = int(round(time_simulation / dt))
    gate_index = start_waypoint_index % len(square_world)
    elapsed_steps = 0
    gates_passed = 0
    total_states: list[np.ndarray] = []
    total_actions: list[np.ndarray] = []
    target_trace: list[np.ndarray] = []
    input_labels = config_model["dataset"]["input_labels"]
    use_sequencing = bool(config_model.get("sequencing", {}).get("value", False))
    seq_len = int(config_model.get("sequencing", {}).get("seq_len", 1))

    initial_distance = float(np.linalg.norm(standby_enu - square_enu[gate_index]))
    print(f"Initial ENU position STDBY: {standby_enu}")
    print(f"First target NN_SQ_{gate_index + 1}: {square_enu[gate_index]}")
    print(f"Initial distance: {initial_distance:.3f} m")
    hover = "n/a" if dynamics_info.hover_omega is None else f"{dynamics_info.hover_omega:.3f} RPM"
    u_hover = "n/a" if dynamics_info.u_hover is None else f"{dynamics_info.u_hover:.6f}"
    print(f"Dynamics model: {dynamics_info.name} | hover={hover} | u_hover={u_hover}")
    print(f"Reset each waypoint: {reset_each_waypoint}")
    print(f"Normalization limits: {normalization_limits}")
    if input_noise_std:
        print(f"Input noise std: {input_noise_std} | seed={input_noise_seed}")

    while elapsed_steps < max_total_steps:
        target = square_world[gate_index]
        relative_world = current_world.copy()
        relative_world[0:3] -= target
        relative_world[8] = _wrap_angle(relative_world[8])
        initial_body = world_to_body_state(relative_world)

        states_body, actions = rollout_checkpoint_controller(
            model=model,
            initial_state=initial_body,
            input_labels=input_labels,
            dt=dt,
            horizon_steps=max_total_steps - elapsed_steps,
            device=device,
            use_sequencing=use_sequencing,
            seq_len=seq_len,
            normalization_limits=normalization_limits,
            force_timespans=False,
            integration_method=integration_method,
            implicit_iters=implicit_iters,
            stop_fn=lambda state, step: np.linalg.norm(state[0:3]) < dist_error,
            input_noise_std=input_noise_std,
            rng=input_noise_rng,
        )

        states_world = body_to_world_trajectory(states_body)
        states_world[:, 0:3] += target

        if total_states:
            total_states.append(states_world[1:])
            if actions.size:
                target_trace.extend([target.copy()] * max(len(states_world) - 1, 0))
        else:
            total_states.append(states_world)
            target_trace.extend([target.copy()] * len(states_world))
        if actions.size:
            total_actions.append(actions)

        current_world = states_world[-1].copy()
        elapsed_steps += max(len(states_world) - 1, 0)
        reached_target = float(np.linalg.norm(current_world[0:3] - target)) < dist_error
        if not reached_target:
            break
        gates_passed += 1
        gate_index = (gate_index + 1) % len(square_world)
        if reset_each_waypoint:
            model = build_lightning_model(config_model, str(checkpoint_path), project_root, device)

    states = np.vstack(total_states)
    actions = np.vstack(total_actions) if total_actions else np.zeros((0, 4), dtype=np.float64)
    targets = np.asarray(target_trace[: len(states)], dtype=np.float64)
    return states, actions, targets, gates_passed


def _plot_actions(actions: np.ndarray, dt: float, output_path: Path, title: str) -> None:
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if actions.size == 0:
        raise ValueError("No actions were generated by the simulation.")

    output_stem = output_path.with_suffix("") if output_path.suffix else output_path
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    csv_path = output_stem.with_suffix(".csv")

    dynamics_info = get_dynamics_info()
    actions01 = np.asarray(actions, dtype=np.float64)
    rpm = dynamics_info.omega_min + np.clip(actions01, 0.0, 1.0) * (
        dynamics_info.omega_max - dynamics_info.omega_min
    )
    time = np.arange(actions01.shape[0], dtype=np.float64) * dt

    if len(actions01) >= 2:
        delta_cmd = np.diff(actions01, axis=0)
        delta_rpm = np.diff(rpm, axis=0)
        stats = []
        for idx in range(4):
            stats.append(
                f"u{idx + 1}: max_delta={np.max(np.abs(delta_cmd[:, idx])):.4f} "
                f"({np.max(np.abs(delta_rpm[:, idx])):.1f} rpm), "
                f"mean_delta={np.mean(np.abs(delta_cmd[:, idx])):.4f} "
                f"({np.mean(np.abs(delta_rpm[:, idx])):.1f} rpm)"
            )
        print("Episode 1 action smoothness | " + " | ".join(stats))

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for motor_idx in range(4):
        axes[motor_idx].plot(
            time,
            rpm[:, motor_idx],
            color=colors[motor_idx],
            alpha=0.95,
            linewidth=1.1,
            label=f"motor {motor_idx + 1}",
        )
    for motor_idx, ax in enumerate(axes):
        ax.set_ylabel(f"motor {motor_idx + 1}\n[rpm]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    delta_rpm = np.vstack([np.zeros((1, 4), dtype=np.float64), np.diff(rpm, axis=0)])
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
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
        for step_idx, (time_s, rpm_row, delta_row) in enumerate(zip(time, rpm, delta_rpm, strict=True)):
            writer.writerow([1, step_idx, time_s, *rpm_row, *delta_row])

    print(f"Action plot saved to {png_path}")
    print(f"Action CSV saved to {csv_path}")


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate the square waypoint path with a checkpoint-backed model.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model", default="NOVA_VERSAOZE", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument(
        "--normalization-limits",
        default=None,
        help="Named profile: bebop1, bebop2_tau_0_06, or bebop2_tau_0_03.",
    )
    parser.add_argument(
        "--normalization-module",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dynamics-model", default="quadrotor_sim_matlab", choices=available_dynamics_models())
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--time-simulation", type=float, default=60.0)
    parser.add_argument("--dist-error", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--input-noise-p-sigma", type=float, default=0.0, help="Gaussian input noise sigma for p [rad/s].")
    parser.add_argument("--input-noise-q-sigma", type=float, default=0.0, help="Gaussian input noise sigma for q [rad/s].")
    parser.add_argument("--input-noise-r-sigma", type=float, default=0.0, help="Gaussian input noise sigma for r [rad/s].")
    parser.add_argument("--input-noise-seed", type=int, default=None)
    parser.add_argument("--integration-method", default="rk4")
    parser.add_argument("--implicit-iters", type=int, default=1)
    parser.add_argument("--start-waypoint-index", type=int, default=3)
    parser.add_argument("--start-alt", type=float, default=1.0)
    parser.add_argument("--waypoint-alt", type=float, default=1.5)
    parser.add_argument("--reset-each-waypoint", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--plot-actions", action="store_true")
    parser.add_argument("--plot-signals", action="store_true")
    parser.add_argument(
        "--action-plot-output",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "square_python_actions.png",
    )
    parser.add_argument(
        "--signals-plot-output",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "square_python_all_state_commands.png",
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", default="gazebo_square_checkpoint.mp4")
    parser.add_argument("--auto-play", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(cli_args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)
    if args.normalization_limits is not None and args.normalization_module is not None:
        raise ValueError("Use either --normalization-limits or legacy --normalization-module, not both.")
    requested_normalization = args.normalization_limits or args.normalization_module
    checkpoint_path, model_config, normalization_limits = _resolve_model(
        args.project_root,
        args.model,
        args.checkpoint,
        args.model_config,
        requested_normalization,
    )
    config_model = load_yaml(model_config)
    input_noise_std = {
        "p": args.input_noise_p_sigma,
        "q": args.input_noise_q_sigma,
        "r": args.input_noise_r_sigma,
    }
    input_noise_std = {key: value for key, value in input_noise_std.items() if value > 0.0}
    standby_enu, square_enu = _square_waypoints(start_alt_m=args.start_alt, waypoint_alt_m=args.waypoint_alt)
    states_world, actions, targets_world, gates_passed = simulate_gazebo_square_python(
        config_model=config_model,
        checkpoint_path=checkpoint_path,
        project_root=args.project_root,
        dt=args.dt,
        time_simulation=args.time_simulation,
        dist_error=args.dist_error,
        integration_method=args.integration_method,
        implicit_iters=args.implicit_iters,
        start_waypoint_index=args.start_waypoint_index,
        reset_each_waypoint=args.reset_each_waypoint,
        start_alt_m=args.start_alt,
        waypoint_alt_m=args.waypoint_alt,
        dynamics_model=args.dynamics_model,
        normalization_limits=normalization_limits,
        tau=args.tau,
        device_name=args.device,
        input_noise_std=input_noise_std or None,
        input_noise_seed=args.input_noise_seed,
    )

    total_time = max(len(states_world) - 1, 0) * args.dt
    total_energy = float(actions.sum() * args.dt) if actions.size else 0.0
    final_enu = _network_world_to_enu(states_world[-1, 0:3])
    print(f"Gates passed: {gates_passed}")
    print(f"Total time: {total_time:.3f} s")
    print(f"Total energy: {total_energy:.3f}")
    print(f"Final ENU position: {final_enu}")
    print(f"Square waypoints ENU: {square_enu}")
    print(f"Standby ENU: {standby_enu}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model config: {model_config}")
    print(f"Dynamics model: {get_dynamics_info().name}")
    if input_noise_std:
        print(f"Input noise std: {input_noise_std} | seed={args.input_noise_seed}")

    if args.plot_actions:
        _plot_actions(
            actions=actions,
            dt=args.dt,
            output_path=args.action_plot_output,
            title=(
                f"square checkpoint actions | dynamics={get_dynamics_info().name} "
                f"tau={get_dynamics_info().tau:.6g}s dist_error={args.dist_error:g} "
                f"input_noise={input_noise_std or {}}"
            ),
        )

    if args.plot_signals:
        _plot_all_signals(
            states_world=states_world,
            actions=actions,
            dt=args.dt,
            output_path=args.signals_plot_output,
            title=(
                f"square checkpoint all signals | dynamics={get_dynamics_info().name} "
                f"tau={get_dynamics_info().tau:.6g}s dist_error={args.dist_error:g} "
                f"input_noise={input_noise_std or {}}"
            ),
        )
        print(f"Signals plot saved to {args.signals_plot_output}")

    if not args.no_animation:
        _animate_square(
            states_world=states_world,
            actions=actions,
            targets_world=targets_world,
            square_enu=square_enu,
            dt=args.dt,
            record=args.record,
            output=args.output,
            auto_play=args.auto_play,
        )


if __name__ == "__main__":
    main()
