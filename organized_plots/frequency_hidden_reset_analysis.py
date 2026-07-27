#!/usr/bin/env python3
"""Compare CfC hidden-state handling and RPM behavior across frequencies."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for import_root in (PROJECT_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simulators.Simulator_gazebo_square_C import (  # noqa: E402
    DEFAULT_INITIAL_MOTOR_RPM,
    _enu_to_network_world,
    _network_world_to_animation,
    _override_dynamics_tau,
    _square_waypoints,
    _wrap_angle,
)
from utils.animation import animate  # noqa: E402
from utils.config import load_yaml  # noqa: E402
from utils.dynamics_models import get_dynamics_info, set_dynamics_model  # noqa: E402
from utils.quadrotor_sim import (  # noqa: E402
    build_input_vector,
    build_lightning_model,
    input_labels_without_time,
    integrate_state,
    make_model_observation,
    normalize_input,
    world_to_body_state,
    body_to_world_state,
)


FREQUENCIES_HZ = (50, 100, 200)
TIME_SIMULATION_S = 20.0
DIST_ERROR_M = 0.001
MOTOR_TAU_S = 0.06


@dataclass(frozen=True)
class ModelSpec:
    name: str
    short_name: str
    checkpoint: Path
    config: Path
    dynamics_model: str
    normalization_limits: str


@dataclass
class Rollout:
    frequency_hz: int
    dt: float
    time: np.ndarray
    actions: np.ndarray
    rpm: np.ndarray
    hidden_stats: np.ndarray
    states_world: np.ndarray
    targets_world: np.ndarray
    gates_passed: int
    completed_steps: int
    finite: bool


MODEL_SPECS = (
    ModelSpec(
        name="conv_cfc_default_n64_bebop2_baseline_correct_dataset",
        short_name="bebop2_correct_dataset",
        checkpoint=PROJECT_ROOT
        / "checkpoints/bebop2/conv_cfc_default_n64_bebop2_baseline_correct_dataset.ckpt",
        config=PROJECT_ROOT
        / "configs/bebop2/conv_cfc_default_n64_bebop2_baseline_correct_dataset.yaml",
        dynamics_model="quadrotor_sim_matlab",
        normalization_limits="bebop2_tau_0_06",
    ),
    ModelSpec(
        name="conv_cfc_default_n64_bebop2_baseline_correct_dataset_no_dt",
        short_name="bebop2_correct_dataset_no_dt",
        checkpoint=PROJECT_ROOT
        / "checkpoints/bebop2/conv_cfc_default_n64_bebop2_baseline_correct_dataset_no_dt.ckpt",
        config=PROJECT_ROOT
        / "configs/bebop2/conv_cfc_default_n64_bebop2_baseline_correct_dataset_no_dt.yaml",
        dynamics_model="quadrotor_sim_matlab",
        normalization_limits="bebop2_tau_0_06",
    ),
    ModelSpec(
        name="bebop1/new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142",
        short_name="bebop1_new_CFC_64_epoch18",
        checkpoint=PROJECT_ROOT
        / "checkpoints/bebop1/new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.ckpt",
        config=PROJECT_ROOT
        / "configs/bebop1/new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.yaml",
        dynamics_model="quadrotor_sim_original",
        normalization_limits="bebop1",
    ),
)


def _hidden_vector(hidden) -> np.ndarray:
    tensors = hidden if isinstance(hidden, tuple) else (hidden,)
    arrays = [
        tensor.detach().cpu().numpy().reshape(-1)
        for tensor in tensors
        if tensor is not None
    ]
    if not arrays:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(arrays).astype(np.float64, copy=False)


def _scale_hidden(hidden, factor: float):
    """Scale every recurrent-state tensor while preserving its container type."""
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple(
            None if tensor is None else tensor * factor
            for tensor in hidden
        )
    return hidden * factor


def _decrement_hidden(hidden, decrement: float):
    """Subtract a constant from every recurrent-state element."""
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple(
            None if tensor is None else tensor - decrement
            for tensor in hidden
        )
    return hidden - decrement


def _initial_world_state(dynamics_name: str) -> tuple[np.ndarray, np.ndarray]:
    standby_enu, square_enu = _square_waypoints(start_alt_m=1.0, waypoint_alt_m=1.5)
    state = np.zeros(19, dtype=np.float64)
    state[0:3] = _enu_to_network_world(standby_enu)
    info = get_dynamics_info()
    initial_rpm = DEFAULT_INITIAL_MOTOR_RPM if dynamics_name == "quadrotor_sim_matlab" else info.omega_mid
    state[15:19] = initial_rpm
    targets = np.asarray([_enu_to_network_world(point) for point in square_enu], dtype=np.float64)
    return state, targets


def run_rollout(
    spec: ModelSpec,
    frequency_hz: int,
    *,
    reset_hidden_each_step: bool,
    collect_hidden: bool,
    device: torch.device,
    hidden_retention: float = 1.0,
    hidden_decrement: float = 0.0,
    start_at_target: bool = False,
    initial_rpm_at_hover: bool = False,
    fixed_target: bool = False,
) -> Rollout:
    dt = 1.0 / float(frequency_hz)
    set_dynamics_model(spec.dynamics_model)
    _override_dynamics_tau(MOTOR_TAU_S)
    info = get_dynamics_info()

    config = load_yaml(spec.config)
    model = build_lightning_model(config, str(spec.checkpoint), PROJECT_ROOT, device)
    input_labels = config["dataset"]["input_labels"]
    base_labels = input_labels_without_time(input_labels)
    use_sequencing = bool(config.get("sequencing", {}).get("value", False))
    seq_len = int(config.get("sequencing", {}).get("seq_len", 1))
    if seq_len != 1:
        raise ValueError(f"{spec.name}: this analysis expects seq_len=1, got {seq_len}")

    state_world, square_world = _initial_world_state(info.name)
    gate_index = 3
    if start_at_target:
        state_world[0:3] = square_world[gate_index]
    if initial_rpm_at_hover:
        hover_rpm = info.hover_omega if info.hover_omega is not None else info.omega_mid
        state_world[15:19] = hover_rpm
    gates_passed = 0
    recurrent_hidden = None
    previous_derivative = None
    actions: list[np.ndarray] = []
    hidden_stats: list[np.ndarray] = []
    states_world: list[np.ndarray] = []
    targets_world: list[np.ndarray] = []
    finite = True
    max_steps = int(round(TIME_SIMULATION_S / dt))
    dt_tensor = (
        torch.tensor(dt, dtype=torch.float32, device=device).reshape(1, 1, 1)
        if getattr(model, "with_time", False)
        else None
    )

    for _step in range(max_steps):
        target = square_world[gate_index]
        states_world.append(state_world.copy())
        targets_world.append(target.copy())
        relative_world = state_world.copy()
        relative_world[0:3] -= target
        relative_world[8] = _wrap_angle(relative_world[8])
        state_body = world_to_body_state(relative_world)

        observation = normalize_input(
            build_input_vector(state_body, base_labels),
            base_labels,
            spec.normalization_limits,
        )
        window = observation.reshape(1, -1)
        observation_tensor = make_model_observation(
            window,
            use_sequencing=use_sequencing,
            device=device,
        )
        input_hidden = None if reset_hidden_each_step else recurrent_hidden
        with torch.no_grad():
            output = model(observation_tensor, hx=input_hidden, timespans=dt_tensor)
            prediction, output_hidden = output if isinstance(output, tuple) else (output, None)

        action = (
            torch.clamp(prediction, min=0.0, max=1.0)
            .squeeze(0)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        if action.shape != (4,):
            raise ValueError(f"{spec.name}: unexpected action shape {action.shape}")
        actions.append(action)

        recurrent_hidden = (
            None
            if reset_hidden_each_step
            else _decrement_hidden(
                _scale_hidden(output_hidden, hidden_retention),
                hidden_decrement,
            )
        )

        if collect_hidden:
            hidden = _hidden_vector(recurrent_hidden)
            if hidden.size == 0:
                raise ValueError(f"{spec.name}: model did not return a recurrent hidden state")
            hidden_stats.append(
                np.asarray(
                    [hidden.min(), hidden.max(), hidden.mean(), hidden.std()],
                    dtype=np.float64,
                )
            )

        next_body, previous_derivative = integrate_state(
            "rk4",
            state_body,
            action,
            dt,
            prev_deriv=previous_derivative,
            implicit_iters=1,
        )
        next_relative_world = body_to_world_state(next_body)
        state_world = next_relative_world
        state_world[0:3] += target

        if not np.all(np.isfinite(state_world)):
            finite = False
            break

        if float(np.linalg.norm(state_world[0:3] - target)) < DIST_ERROR_M:
            if not fixed_target:
                gates_passed += 1
                gate_index = (gate_index + 1) % len(square_world)

    action_array = np.asarray(actions, dtype=np.float64)
    rpm = info.omega_min + action_array * (info.omega_max - info.omega_min)
    hidden_array = (
        np.asarray(hidden_stats, dtype=np.float64)
        if hidden_stats
        else np.zeros((len(action_array), 4), dtype=np.float64)
    )
    time = np.arange(len(action_array), dtype=np.float64) * dt
    return Rollout(
        frequency_hz=frequency_hz,
        dt=dt,
        time=time,
        actions=action_array,
        rpm=rpm,
        hidden_stats=hidden_array,
        states_world=np.asarray(states_world, dtype=np.float64),
        targets_world=np.asarray(targets_world, dtype=np.float64),
        gates_passed=gates_passed,
        completed_steps=len(action_array),
        finite=finite,
    )


def _plot_hidden(
    spec: ModelSpec,
    rollouts: list[Rollout],
    output: Path,
    *,
    hidden_state_description: str = "recurrent state preserved",
) -> None:
    statistics = (
        (1, "max", "#b2182b"),
        (0, "min", "#2166ac"),
        (3, "std", "#762a83"),
        (2, "mean", "#1b7837"),
    )
    fig, axes = plt.subplots(
        4,
        len(rollouts),
        figsize=(18, 12),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    for column, rollout in enumerate(rollouts):
        for row, (stat_index, stat_name, color) in enumerate(statistics):
            axis = axes[row, column]
            axis.plot(
                rollout.time,
                rollout.hidden_stats[:, stat_index],
                color=color,
                linewidth=1.0,
            )
            axis.grid(True, alpha=0.25)
            if column == 0:
                axis.set_ylabel(f"hidden {stat_name}")
            if row == 0:
                axis.set_title(
                    f"{rollout.frequency_hz} Hz — dt={rollout.dt:g} s\n"
                    f"gates={rollout.gates_passed}"
                )
            if row == len(statistics) - 1:
                axis.set_xlabel("time [s]")
    fig.suptitle(
        f"{spec.name}\n"
        f"CfC hidden-state evolution — {hidden_state_description}\n"
        f"time=20 s | dist_error=0.001 m | tau=0.06 s | no noise | "
        f"dynamics={spec.dynamics_model} | normalization={spec.normalization_limits}",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_rpm(
    spec: ModelSpec,
    rollouts: list[Rollout],
    output: Path,
    *,
    hidden_state_description: str,
) -> None:
    colors = ("#2166ac", "#b2182b", "#1b7837", "#762a83")
    fig, axes = plt.subplots(
        4,
        len(rollouts),
        figsize=(18, 12),
        sharex="col",
        sharey=True,
        constrained_layout=True,
    )
    for column, rollout in enumerate(rollouts):
        for motor in range(4):
            axis = axes[motor, column]
            axis.plot(
                rollout.time,
                rollout.rpm[:, motor],
                color=colors[motor],
                linewidth=0.9,
                label=f"motor {motor + 1}",
            )
            axis.set_ylim(4900.0, 10100.0)
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper right", fontsize=8)
            if column == 0:
                axis.set_ylabel(f"motor {motor + 1} [RPM]")
            if motor == 0:
                axis.set_title(
                    f"{rollout.frequency_hz} Hz — dt={rollout.dt:g} s\n"
                    f"gates={rollout.gates_passed}"
                )
            if motor == 3:
                axis.set_xlabel("time [s]")
    fig.suptitle(
        f"{spec.name}\n"
        f"Motor RPM commands — {hidden_state_description}\n"
        f"time=20 s | dist_error=0.001 m | tau=0.06 s | no noise | "
        f"dynamics={spec.dynamics_model} | normalization={spec.normalization_limits}",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _write_summary(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir: Path, device_name: str) -> list[Path]:
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    generated: list[Path] = []
    summary_rows: list[dict[str, object]] = []

    for spec in MODEL_SPECS:
        hidden_rollouts = [
            run_rollout(
                spec,
                frequency,
                reset_hidden_each_step=False,
                collect_hidden=True,
                device=device,
            )
            for frequency in FREQUENCIES_HZ
        ]
        reset_rollouts = [
            run_rollout(
                spec,
                frequency,
                reset_hidden_each_step=True,
                collect_hidden=False,
                device=device,
            )
            for frequency in FREQUENCIES_HZ
        ]

        stem = _safe_stem(spec.short_name)
        hidden_path = output_dir / f"{stem}_hidden_stats_50_100_200Hz.png"
        rpm_path = output_dir / f"{stem}_rpm_reset_each_timestep_50_100_200Hz.png"
        rpm_preserved_path = output_dir / f"{stem}_rpm_hidden_preserved_50_100_200Hz.png"
        _plot_hidden(spec, hidden_rollouts, hidden_path)
        _plot_rpm(
            spec,
            reset_rollouts,
            rpm_path,
            hidden_state_description="CfC hidden state reset before every timestep",
        )
        _plot_rpm(
            spec,
            hidden_rollouts,
            rpm_preserved_path,
            hidden_state_description="CfC hidden state preserved continuously",
        )
        generated.extend((hidden_path, rpm_path, rpm_preserved_path))

        for mode, rollouts in (
            ("hidden_recurrent", hidden_rollouts),
            ("rpm_reset_each_timestep", reset_rollouts),
        ):
            for rollout in rollouts:
                summary_rows.append(
                    {
                        "model": spec.name,
                        "mode": mode,
                        "frequency_hz": rollout.frequency_hz,
                        "dt_s": rollout.dt,
                        "requested_time_s": TIME_SIMULATION_S,
                        "completed_time_s": rollout.completed_steps * rollout.dt,
                        "completed_steps": rollout.completed_steps,
                        "gates_passed": rollout.gates_passed,
                        "finite": rollout.finite,
                        "tau_s": MOTOR_TAU_S,
                        "dist_error_m": DIST_ERROR_M,
                        "noise": "off",
                        "dynamics_model": spec.dynamics_model,
                        "normalization_limits": spec.normalization_limits,
                        "rpm_min": float(np.min(rollout.rpm)),
                        "rpm_max": float(np.max(rollout.rpm)),
                    }
                )
        print(
            f"{spec.name}: saved {hidden_path.name}, {rpm_path.name}, "
            f"and {rpm_preserved_path.name}"
        )

    _write_summary(summary_rows, output_dir / "run_summary.csv")
    return generated


def run_retention_analysis(
    output_dir: Path,
    device_name: str,
    hidden_retention: float,
) -> list[Path]:
    if not 0.0 <= hidden_retention <= 1.0:
        raise ValueError("hidden_retention must be between 0 and 1")

    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    spec = MODEL_SPECS[0]
    rollouts = [
        run_rollout(
            spec,
            frequency,
            reset_hidden_each_step=False,
            collect_hidden=True,
            device=device,
            hidden_retention=hidden_retention,
        )
        for frequency in FREQUENCIES_HZ
    ]

    retention_stem = f"{hidden_retention:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    stem = _safe_stem(spec.short_name)
    hidden_path = (
        output_dir
        / f"{stem}_hidden_stats_retention_{retention_stem}_50_100_200Hz.png"
    )
    rpm_path = (
        output_dir
        / f"{stem}_rpm_hidden_retention_{retention_stem}_50_100_200Hz.png"
    )
    description = f"recurrent state multiplied by {hidden_retention:g} after every timestep"
    _plot_hidden(
        spec,
        rollouts,
        hidden_path,
        hidden_state_description=description,
    )
    _plot_rpm(
        spec,
        rollouts,
        rpm_path,
        hidden_state_description=f"CfC hidden state × {hidden_retention:g} after every timestep",
    )

    summary_rows = []
    for rollout in rollouts:
        summary_rows.append(
            {
                "model": spec.name,
                "mode": f"hidden_retention_{hidden_retention:g}",
                "hidden_retention": hidden_retention,
                "frequency_hz": rollout.frequency_hz,
                "dt_s": rollout.dt,
                "requested_time_s": TIME_SIMULATION_S,
                "completed_time_s": rollout.completed_steps * rollout.dt,
                "completed_steps": rollout.completed_steps,
                "gates_passed": rollout.gates_passed,
                "finite": rollout.finite,
                "tau_s": MOTOR_TAU_S,
                "dist_error_m": DIST_ERROR_M,
                "noise": "off",
                "dynamics_model": spec.dynamics_model,
                "normalization_limits": spec.normalization_limits,
                "rpm_min": float(np.min(rollout.rpm)),
                "rpm_max": float(np.max(rollout.rpm)),
            }
        )
    _write_summary(
        summary_rows,
        output_dir / f"{stem}_retention_{retention_stem}_summary.csv",
    )
    return [hidden_path, rpm_path]


def run_decrement_analysis(
    output_dir: Path,
    device_name: str,
    hidden_decrement: float,
) -> list[Path]:
    if hidden_decrement < 0.0:
        raise ValueError("hidden_decrement must be non-negative")

    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    spec = MODEL_SPECS[0]
    rollouts = [
        run_rollout(
            spec,
            frequency,
            reset_hidden_each_step=False,
            collect_hidden=True,
            device=device,
            hidden_decrement=hidden_decrement,
        )
        for frequency in FREQUENCIES_HZ
    ]

    decrement_stem = f"{hidden_decrement:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    stem = _safe_stem(spec.short_name)
    hidden_path = (
        output_dir
        / f"{stem}_hidden_stats_decrement_{decrement_stem}_50_100_200Hz.png"
    )
    rpm_path = (
        output_dir
        / f"{stem}_rpm_hidden_decrement_{decrement_stem}_50_100_200Hz.png"
    )
    description = (
        f"{hidden_decrement:g} subtracted from recurrent state after every timestep"
    )
    _plot_hidden(
        spec,
        rollouts,
        hidden_path,
        hidden_state_description=description,
    )
    _plot_rpm(
        spec,
        rollouts,
        rpm_path,
        hidden_state_description=(
            f"{hidden_decrement:g} subtracted from CfC hidden state after every timestep"
        ),
    )

    summary_rows = []
    for rollout in rollouts:
        summary_rows.append(
            {
                "model": spec.name,
                "mode": f"hidden_decrement_{hidden_decrement:g}",
                "hidden_decrement": hidden_decrement,
                "frequency_hz": rollout.frequency_hz,
                "dt_s": rollout.dt,
                "requested_time_s": TIME_SIMULATION_S,
                "completed_time_s": rollout.completed_steps * rollout.dt,
                "completed_steps": rollout.completed_steps,
                "gates_passed": rollout.gates_passed,
                "finite": rollout.finite,
                "tau_s": MOTOR_TAU_S,
                "dist_error_m": DIST_ERROR_M,
                "noise": "off",
                "dynamics_model": spec.dynamics_model,
                "normalization_limits": spec.normalization_limits,
                "rpm_min": float(np.min(rollout.rpm)),
                "rpm_max": float(np.max(rollout.rpm)),
            }
        )
    _write_summary(
        summary_rows,
        output_dir / f"{stem}_decrement_{decrement_stem}_summary.csv",
    )
    return [hidden_path, rpm_path]


def animate_decrement_rollout(
    device_name: str,
    hidden_decrement: float,
    frequency_hz: int,
) -> None:
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    spec = MODEL_SPECS[0]
    rollout = run_rollout(
        spec,
        frequency_hz,
        reset_hidden_each_step=False,
        collect_hidden=True,
        device=device,
        hidden_decrement=hidden_decrement,
    )
    states_anim = _network_world_to_animation(rollout.states_world[:, 0:3])
    targets_anim = _network_world_to_animation(rollout.targets_world)
    _, square_enu = _square_waypoints(start_alt_m=1.0, waypoint_alt_m=1.5)
    waypoints_anim = _network_world_to_animation(
        np.asarray([_enu_to_network_world(wp) for wp in square_enu])
    )
    animate(
        t=rollout.time,
        x=states_anim[:, 0],
        y=states_anim[:, 1],
        z=states_anim[:, 2],
        phi=rollout.states_world[:, 6],
        theta=rollout.states_world[:, 7],
        psi=rollout.states_world[:, 8],
        u=rollout.actions,
        target=targets_anim,
        waypoints=waypoints_anim,
        record=False,
        auto_play=True,
        close_on_end=False,
        draw_path=True,
    )


def run_fixed_waypoint_hover_analysis(
    output_dir: Path,
    device_name: str,
    specs: tuple[ModelSpec, ...] | None = None,
) -> list[Path]:
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    selected_specs = specs if specs is not None else (MODEL_SPECS[0],)
    generated: list[Path] = []
    for spec in selected_specs:
        rollouts = [
            run_rollout(
                spec,
                frequency,
                reset_hidden_each_step=False,
                collect_hidden=True,
                device=device,
                start_at_target=True,
                initial_rpm_at_hover=True,
                fixed_target=True,
            )
            for frequency in FREQUENCIES_HZ
        ]

        stem = _safe_stem(spec.short_name)
        hidden_path = (
            output_dir
            / f"{stem}_hidden_stats_fixed_waypoint_hover_50_100_200Hz.png"
        )
        rpm_path = (
            output_dir
            / f"{stem}_rpm_fixed_waypoint_hover_50_100_200Hz.png"
        )
        description = (
            "state preserved; starts at fixed waypoint with all motors at hover RPM"
        )
        _plot_hidden(
            spec,
            rollouts,
            hidden_path,
            hidden_state_description=description,
        )
        _plot_rpm(
            spec,
            rollouts,
            rpm_path,
            hidden_state_description=(
                "CfC state preserved; initial pose=fixed waypoint; "
                "initial motors=hover RPM"
            ),
        )

        info = get_dynamics_info()
        hover_rpm = (
            info.hover_omega if info.hover_omega is not None else info.omega_mid
        )
        summary_rows = []
        for rollout in rollouts:
            summary_rows.append(
                {
                    "model": spec.name,
                    "mode": "fixed_waypoint_hover_hidden_preserved",
                    "frequency_hz": rollout.frequency_hz,
                    "dt_s": rollout.dt,
                    "requested_time_s": TIME_SIMULATION_S,
                    "completed_time_s": rollout.completed_steps * rollout.dt,
                    "completed_steps": rollout.completed_steps,
                    "finite": rollout.finite,
                    "initial_position": "selected_waypoint",
                    "fixed_waypoint": True,
                    "initial_motor_rpm": hover_rpm,
                    "hidden_reset_each_step": False,
                    "hidden_decrement": 0.0,
                    "hidden_retention": 1.0,
                    "tau_s": MOTOR_TAU_S,
                    "dist_error_m": DIST_ERROR_M,
                    "noise": "off",
                    "dynamics_model": spec.dynamics_model,
                    "normalization_limits": spec.normalization_limits,
                    "rpm_min": float(np.min(rollout.rpm)),
                    "rpm_max": float(np.max(rollout.rpm)),
                }
            )
        _write_summary(
            summary_rows,
            output_dir / f"{stem}_fixed_waypoint_hover_summary.csv",
        )
        generated.extend((hidden_path, rpm_path))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "organized_plots/frequency_hidden_reset_20s",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--hidden-retention",
        type=float,
        default=None,
        help=(
            "Generate only the Bebop2-with-dt plots while multiplying the "
            "recurrent state by this factor after every timestep."
        ),
    )
    parser.add_argument(
        "--hidden-decrement",
        type=float,
        default=None,
        help=(
            "Generate only the Bebop2-with-dt plots while subtracting this "
            "value from every recurrent-state element after every timestep."
        ),
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Open the selected hidden-decrement rollout in the interactive viewer.",
    )
    parser.add_argument("--animation-frequency", type=int, default=100)
    parser.add_argument(
        "--animation-decrements",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Open multiple hidden-decrement animations sequentially. "
            "Closing one window opens the next."
        ),
    )
    parser.add_argument(
        "--fixed-waypoint-hover",
        action="store_true",
        help=(
            "Generate Bebop2-with-dt plots starting exactly at a fixed waypoint, "
            "with all motors initialized at hover RPM and hidden state preserved."
        ),
    )
    parser.add_argument(
        "--fixed-waypoint-hover-models",
        nargs="+",
        choices=tuple(spec.short_name for spec in MODEL_SPECS),
        default=None,
        help="Select which model variants to use for --fixed-waypoint-hover.",
    )
    args = parser.parse_args()
    if args.hidden_retention is not None and args.hidden_decrement is not None:
        parser.error("--hidden-retention and --hidden-decrement are mutually exclusive")
    if args.fixed_waypoint_hover:
        selected_specs = (
            tuple(
                spec
                for spec in MODEL_SPECS
                if spec.short_name in args.fixed_waypoint_hover_models
            )
            if args.fixed_waypoint_hover_models is not None
            else None
        )
        generated = run_fixed_waypoint_hover_analysis(
            args.output_dir,
            args.device,
            selected_specs,
        )
    elif args.animation_decrements is not None:
        for decrement in args.animation_decrements:
            print(
                f"Opening animation: hidden decrement={decrement:g}, "
                f"frequency={args.animation_frequency} Hz"
            )
            animate_decrement_rollout(
                args.device,
                decrement,
                args.animation_frequency,
            )
        generated = []
    elif args.animate:
        if args.hidden_decrement is None:
            parser.error("--animate requires --hidden-decrement")
        animate_decrement_rollout(
            args.device,
            args.hidden_decrement,
            args.animation_frequency,
        )
        generated = []
    elif args.hidden_decrement is not None:
        generated = run_decrement_analysis(
            args.output_dir,
            args.device,
            args.hidden_decrement,
        )
    elif args.hidden_retention is not None:
        generated = run_retention_analysis(
            args.output_dir,
            args.device,
            args.hidden_retention,
        )
    else:
        generated = run(args.output_dir, args.device)
    print(f"Generated {len(generated)} figures in {args.output_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
