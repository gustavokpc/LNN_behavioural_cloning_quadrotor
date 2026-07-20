#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Square-waypoint simulation using the exported CFC controller."""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
from typing import Iterable

_RUNS_DIR = Path(__file__).resolve().parent / "runs"
os.environ.setdefault("MPLCONFIGDIR", str(_RUNS_DIR / ".matplotlib"))

import numpy as np

from utils.animation import animate
from utils.c_controller import CController
from utils.config import load_yaml
from utils.dynamics_models import available_dynamics_models, get_dynamics_info, get_dynamics_model, set_dynamics_model
from utils.quadrotor_sim import body_to_world_trajectory, world_to_body_state
from utils.quadrotor_sim_c import rollout_c_controller


MODEL_PRESETS = {
    "MLP": ("configs/bebop1/mlp_epoch=19_val_loss=0.003130.yaml", "C_codes/bebop1/MLP"),
    "LTC": ("configs/bebop1/LTC_64_neurons_seq_1_epoch=18_val_loss=0.000193.yaml", "C_codes/bebop1/LTC"),
    "RNN": ("configs/bebop1/RNN_64_neurons_seq_1_epoch=17_val_loss=0.000147.yaml", "C_codes/bebop1/RNN"),
    "CONV_CFC_DEFAULT": ("configs/bebop1/conv_cfc_default_n64_epoch=17_val_loss=0.000326.yaml", "C_codes/bebop1/CONV_CFC_DEFAULT"),
    "CFC": ("configs/bebop1/new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.yaml", "C_codes/bebop1/CFC"),
    "CFC_PURE": ("configs/bebop1/new_CFC_pure_64_neurons_seq_1_epoch=17_val_loss=0.000203.yaml", "C_codes/bebop1/CFC_PURE"),
    "CTRNN": ("configs/bebop1/new_CTRNN_64_neurons_seq_1_epoch=19_val_loss=0.000150.yaml", "C_codes/bebop1/CTRNN"),
    "GRU": ("configs/bebop1/new_GRU_64_neurons_seq_1_epoch=19_val_loss=0.000088.yaml", "C_codes/bebop1/GRU"),
    "LSTM": ("configs/bebop1/new_LSTM_64_neurons_seq_1_epoch=17_val_loss=0.000092.yaml", "C_codes/bebop1/LSTM"),
    "NCP_CFC": ("configs/bebop1/new_NCP_CFC_60_neurons_seq_1_epoch=18_val_loss=0.000143.yaml", "C_codes/bebop1/NCP_CFC"),
    "NOVA_VERSAOZE": (
        "configs/bebop2/NOVA_VERSAOZE_BEBP2_conv_cfc_default_n64_bebop2_epoch=19_val_loss=0.000098.yaml",
        "C_codes/bebop2/NOVA_VERSAOZE_BEBP2_CONV_CFC",
    ),
}
DEFAULT_STANDBY_ENU = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
DEFAULT_SQUARE_ENU = np.asarray(
    [
        [2.0, 1.5, 1.5],
        [2.0, -1.5, 1.5],
        [-2.0, -1.5, 1.5],
        [-2.0, 1.5, 1.5],
    ],
    dtype=np.float64,
)
DEFAULT_INITIAL_MOTOR_RPM = 7750.0


def resolve_model_paths(project_root: Path, model: str, model_config: Path | None, c_model_dir: Path | None) -> tuple[Path, Path]:
    model_key = model.upper()
    if model_key not in MODEL_PRESETS:
        valid = ", ".join(sorted(MODEL_PRESETS))
        raise KeyError(f"Unknown model '{model}'. Choose one of: {valid}")
    default_config, default_c_dir = MODEL_PRESETS[model_key]
    config_path = model_config or project_root / default_config
    c_dir = c_model_dir or project_root / default_c_dir
    return config_path, c_dir


def _enu_to_network_world(vec: np.ndarray) -> np.ndarray:
    """Match nn_cfc_control.c: ENU -> network world frame {Y, X, -Z}."""
    return np.asarray([vec[1], vec[0], -vec[2]], dtype=np.float64)


def _network_world_to_enu(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.stack([points[..., 1], points[..., 0], -points[..., 2]], axis=-1)


def _network_world_to_animation(points: np.ndarray) -> np.ndarray:
    """Use Paparazzi ENU x/y but keep simulator z sign for the existing animator."""
    points = np.asarray(points, dtype=np.float64)
    return np.stack([points[..., 1], points[..., 0], points[..., 2]], axis=-1)


def _square_waypoints(
    start_alt_m: float = 1.0,
    waypoint_alt_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    standby = DEFAULT_STANDBY_ENU.copy()
    square = DEFAULT_SQUARE_ENU.copy()
    standby[2] = float(start_alt_m)
    square[:, 2] = float(waypoint_alt_m)
    return standby, square


def _wrap_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


def _override_dynamics_tau(tau: float | None) -> None:
    if tau is None:
        return
    if tau <= 0.0:
        raise ValueError("--tau must be positive.")
    model = get_dynamics_model()
    if not hasattr(model, "TAU"):
        raise AttributeError(f"Dynamics model '{model.INFO.name}' does not expose a TAU constant.")
    model.TAU = float(tau)
    model.INFO = dataclasses.replace(
        model.INFO,
        description=f"{model.INFO.description} Runtime motor time constant override tau={tau:.6g}s.",
        tau=float(tau),
    )


def simulate_gazebo_square(
    config_model: dict,
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
    c_model_dir: Path,
    dynamics_model: str,
    tau: float | None = None,
    input_noise_std: dict[str, float] | None = None,
    input_noise_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    set_dynamics_model(dynamics_model)
    _override_dynamics_tau(tau)
    dynamics_info = get_dynamics_info()
    input_noise_rng = np.random.default_rng(input_noise_seed) if input_noise_std else None
    standby_enu, square_enu = _square_waypoints(
        start_alt_m=start_alt_m,
        waypoint_alt_m=waypoint_alt_m,
    )
    square_world = np.asarray([_enu_to_network_world(wp) for wp in square_enu], dtype=np.float64)
    current_world = np.zeros(19, dtype=np.float64)
    current_world[0:3] = _enu_to_network_world(standby_enu)
    initial_motor_rpm = DEFAULT_INITIAL_MOTOR_RPM if dynamics_info.name == "quadrotor_sim_matlab" else dynamics_info.omega_mid
    current_world[15:19] = initial_motor_rpm

    controller = CController(c_model_dir)
    controller.reset()
    max_total_steps = int(round(time_simulation / dt))
    gate_index = start_waypoint_index % len(square_world)
    elapsed_steps = 0
    gates_passed = 0
    total_states: list[np.ndarray] = []
    total_actions: list[np.ndarray] = []
    target_trace: list[np.ndarray] = []
    full_input_labels = config_model["dataset"]["input_labels"]
    input_labels_without_time = [label for label in full_input_labels if label not in {"t", "dt"}]

    def _expanded_input_size(labels: list[str]) -> int:
        return sum(4 if label == "omega" else 1 for label in labels)

    if _expanded_input_size(full_input_labels) == controller.num_states:
        c_input_labels = full_input_labels
    elif _expanded_input_size(input_labels_without_time) == controller.num_states:
        c_input_labels = input_labels_without_time
    else:
        raise ValueError(
            f"C controller expects {controller.num_states} inputs, but config expands to "
            f"{_expanded_input_size(full_input_labels)} with time and "
            f"{_expanded_input_size(input_labels_without_time)} without time."
        )

    initial_distance = float(np.linalg.norm(standby_enu - square_enu[gate_index]))
    print(f"Initial ENU position STDBY: {standby_enu}")
    print(f"First target NN_SQ_{gate_index + 1}: {square_enu[gate_index]}")
    print(f"Initial distance: {initial_distance:.3f} m")
    hover = "n/a" if dynamics_info.hover_omega is None else f"{dynamics_info.hover_omega:.3f} RPM"
    u_hover = "n/a" if dynamics_info.u_hover is None else f"{dynamics_info.u_hover:.6f}"
    print(f"Dynamics model: {dynamics_info.name} | hover={hover} | u_hover={u_hover}")
    print(f"Reset each waypoint: {reset_each_waypoint}")

    while elapsed_steps < max_total_steps:
        target = square_world[gate_index]
        relative_world = current_world.copy()
        relative_world[0:3] -= target
        relative_world[8] = _wrap_angle(relative_world[8])
        initial_body = world_to_body_state(relative_world)

        states_body, actions = rollout_c_controller(
            controller=controller,
            initial_state=initial_body,
            input_labels=c_input_labels,
            dt=dt,
            horizon_steps=max_total_steps - elapsed_steps,
            integration_method=integration_method,
            implicit_iters=implicit_iters,
            stop_fn=lambda state, step: np.linalg.norm(state[0:3]) < dist_error,
            reset_controller=reset_each_waypoint,
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

    states = np.vstack(total_states)
    actions = np.vstack(total_actions) if total_actions else np.zeros((0, 4), dtype=np.float64)
    targets = np.asarray(target_trace[: len(states)], dtype=np.float64)
    return states, actions, targets, gates_passed


def _animate_square(
    states_world: np.ndarray,
    actions: np.ndarray,
    targets_world: np.ndarray,
    square_enu: np.ndarray,
    dt: float,
    record: bool,
    output: str,
    auto_play: bool,
) -> None:
    steps = min(len(states_world), len(actions))
    states_world = states_world[:steps]
    actions = actions[:steps]
    targets_world = targets_world[:steps]
    states_anim = _network_world_to_animation(states_world[:, 0:3])
    targets_anim = _network_world_to_animation(targets_world)
    waypoints_anim = _network_world_to_animation(np.asarray([_enu_to_network_world(wp) for wp in square_enu]))
    t = np.arange(steps, dtype=np.float64) * dt

    animate(
        t=t,
        x=states_anim[:, 0],
        y=states_anim[:, 1],
        z=states_anim[:, 2],
        phi=states_world[:, 6],
        theta=states_world[:, 7],
        psi=states_world[:, 8],
        u=actions,
        target=targets_anim,
        waypoints=waypoints_anim,
        file=output,
        record=record,
        auto_play=auto_play or record,
        close_on_end=record,
        draw_path=True,
    )


def _plot_all_signals(
    states_world: np.ndarray,
    actions: np.ndarray,
    dt: float,
    output_path: Path,
    title: str,
) -> None:
    if actions.size == 0:
        raise ValueError("No actions were generated by the simulation.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = min(len(states_world), len(actions))
    states = states_world[:steps]
    actions01 = np.asarray(actions[:steps], dtype=np.float64)
    time = np.arange(steps, dtype=np.float64) * dt
    dynamics_info = get_dynamics_info()

    pos_enu = _network_world_to_enu(states[:, 0:3])
    vel_enu = _network_world_to_enu(states[:, 3:6])
    rpm_obs = states[:, 15:19]
    rpm_ref = dynamics_info.omega_min + np.clip(actions01, 0.0, 1.0) * (
        dynamics_info.omega_max - dynamics_info.omega_min
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(7, 1, figsize=(18, 22), sharex=True)
    for ax in axes:
        ax.set_facecolor("#fff1df")
        ax.grid(True, alpha=0.25)

    axes[0].plot(time, pos_enu[:, 0], label="pos_x")
    axes[0].plot(time, pos_enu[:, 1], label="pos_y")
    axes[0].plot(time, pos_enu[:, 2], label="pos_z")
    axes[0].set_ylabel("pos [m]")

    axes[1].plot(time, vel_enu[:, 0], label="vel_x")
    axes[1].plot(time, vel_enu[:, 1], label="vel_y")
    axes[1].plot(time, vel_enu[:, 2], label="vel_z")
    axes[1].set_ylabel("vel [m/s]")

    axes[2].plot(time, states[:, 6], label="att_phi")
    axes[2].plot(time, states[:, 7], label="att_theta")
    axes[2].plot(time, states[:, 8], label="att_psi")
    axes[2].set_ylabel("att [rad]")

    axes[3].plot(time, states[:, 9], label="rate_p")
    axes[3].plot(time, states[:, 10], label="rate_q")
    axes[3].plot(time, states[:, 11], label="rate_r")
    axes[3].set_ylabel("rates [rad/s]")

    for idx in range(4):
        axes[4].plot(time, rpm_obs[:, idx], label=f"rpm_obs_{idx + 1}")
    axes[4].set_ylabel("rpm_obs")

    for idx in range(4):
        axes[5].plot(time, rpm_ref[:, idx], label=f"rpm_ref_{idx + 1}")
    axes[5].set_ylabel("rpm_ref")

    for idx in range(4):
        axes[6].plot(time, actions01[:, idx], label=f"cmd_u{idx + 1}")
    axes[6].set_ylabel("cmd [0-1]")
    axes[6].set_xlabel("time [s]")

    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Simulate the square waypoint path with a C-exported model.")
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--model", default="CFC", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--c-model-dir", type=Path, default=None)
    parser.add_argument("--dynamics-model", default="quadrotor_sim_matlab", choices=available_dynamics_models())
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--time-simulation", type=float, default=60.0)
    parser.add_argument("--dist-error", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=None, help="Optional runtime motor time constant override in seconds.")
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
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--plot-actions", action="store_true")
    parser.add_argument("--plot-signals", action="store_true")
    parser.add_argument(
        "--action-plot-output",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "cfc_sl_bebop2_actions.png",
    )
    parser.add_argument(
        "--signals-plot-output",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "square_all_state_commands.png",
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", default="gazebo_square_cfc.mp4")
    parser.add_argument("--auto-play", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(cli_args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)
    model_config, c_model_dir = resolve_model_paths(args.project_root, args.model, args.model_config, args.c_model_dir)
    config_model = load_yaml(model_config)
    input_noise_std = {
        "p": args.input_noise_p_sigma,
        "q": args.input_noise_q_sigma,
        "r": args.input_noise_r_sigma,
    }
    input_noise_std = {key: value for key, value in input_noise_std.items() if value > 0.0}
    standby_enu, square_enu = _square_waypoints(
        start_alt_m=args.start_alt,
        waypoint_alt_m=args.waypoint_alt,
    )
    states_world, actions, targets_world, gates_passed = simulate_gazebo_square(
        config_model=config_model,
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
        c_model_dir=c_model_dir,
        dynamics_model=args.dynamics_model,
        tau=args.tau,
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
    print(f"Dynamics model: {get_dynamics_info().name}")
    print(f"Model config: {model_config}")
    print(f"C model dir: {c_model_dir}")
    if input_noise_std:
        print(f"Input noise std: {input_noise_std} | seed={args.input_noise_seed}")

    if args.plot_actions:
        import csv
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if actions.size == 0:
            raise ValueError("No actions were generated by the simulation.")

        output_stem = args.action_plot_output.with_suffix("") if args.action_plot_output.suffix else args.action_plot_output
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        png_path = output_stem.with_suffix(".png")
        csv_path = output_stem.with_suffix(".csv")

        dynamics_info = get_dynamics_info()
        actions01 = np.asarray(actions, dtype=np.float64)
        rpm = dynamics_info.omega_min + np.clip(actions01, 0.0, 1.0) * (
            dynamics_info.omega_max - dynamics_info.omega_min
        )
        time = np.arange(actions01.shape[0], dtype=np.float64) * args.dt

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
        fig.suptitle(f"square_waypoints CFC actions - {c_model_dir.name}")
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        plt.close(fig)

        delta_rpm = np.vstack([np.zeros((1, 4), dtype=np.float64), np.diff(rpm, axis=0)])
        with csv_path.open("w", newline="") as csv_file:
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

    if args.plot_signals:
        output_path = args.signals_plot_output
        _plot_all_signals(
            states_world=states_world,
            actions=actions,
            dt=args.dt,
            output_path=output_path,
            title=(
                f"square CFC all signals | dynamics={get_dynamics_info().name} "
                f"tau={get_dynamics_info().tau:.6g}s dist_error={args.dist_error:g} "
                f"input_noise={input_noise_std or {}}"
            ),
        )
        print(f"Signals plot saved to {output_path}")

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
