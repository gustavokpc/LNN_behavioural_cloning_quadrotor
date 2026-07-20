#!/usr/bin/env python3
"""Replay recorded dataset motor commands through a selectable quadrotor dynamics model."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from LNN_behavioural_cloning_quadrotor.utils.dynamics_models import available_dynamics_models, load_dynamics_model


DYNAMICS_MODEL = load_dynamics_model("quadrotor_sim_matlab")


def _set_dynamics_model(name: str) -> None:
    global DYNAMICS_MODEL
    DYNAMICS_MODEL = load_dynamics_model(name)


STATE_LABELS = [
    "dx", "dy", "dz",
    "vx", "vy", "vz",
    "phi", "theta", "psi",
    "p", "q", "r",
    "Mx_ext", "My_ext", "Mz_ext",
    "omega1", "omega2", "omega3", "omega4",
]


STATE_DATASET_KEYS = [
    "dx", "dy", "dz",
    "vx", "vy", "vz",
    "phi", "theta", "psi",
    "p", "q", "r",
]
EXT_MOMENT_KEYS = ["Mx_ext", "My_ext", "Mz_ext"]


def _rotation_body_to_world(phi: float, theta: float, psi: float) -> np.ndarray:
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]])
    ry = np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]])
    rz = np.array([[np.cos(psi), -np.sin(psi), 0.0], [np.sin(psi), np.cos(psi), 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _body_to_world_state(state_body: np.ndarray) -> np.ndarray:
    state_world = np.asarray(state_body, dtype=np.float64).copy()
    dx, dy, dz, vx, vy, vz, phi, theta, psi = state_world[:9]
    rotation = _rotation_body_to_world(phi, theta, psi)
    state_world[0:3] = -rotation @ np.array([dx, dy, dz])
    state_world[3:6] = rotation @ np.array([vx, vy, vz])
    return state_world


def _body_to_world_trajectory(states_body: np.ndarray) -> np.ndarray:
    return np.asarray([_body_to_world_state(state) for state in states_body], dtype=np.float64)


def _integrate_state(
    method: str,
    state: np.ndarray,
    action: np.ndarray,
    dt: float,
    prev_deriv: np.ndarray | None = None,
    implicit_iters: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    method = method.lower()
    deriv = DYNAMICS_MODEL.dynamics(state, action)

    if method in {"explicit", "euler"}:
        return state + dt * deriv, deriv

    if method == "rk4":
        k1 = deriv
        k2 = DYNAMICS_MODEL.dynamics(state + 0.5 * dt * k1, action)
        k3 = DYNAMICS_MODEL.dynamics(state + 0.5 * dt * k2, action)
        k4 = DYNAMICS_MODEL.dynamics(state + dt * k3, action)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), deriv

    if method == "adams-bashforth":
        if prev_deriv is None:
            return state + dt * deriv, deriv
        return state + dt * (1.5 * deriv - 0.5 * prev_deriv), deriv

    if method == "adams-moulton":
        if prev_deriv is None:
            return state + dt * deriv, deriv
        predictor = state + dt * (1.5 * deriv - 0.5 * prev_deriv)
        return state + 0.5 * dt * (DYNAMICS_MODEL.dynamics(predictor, action) + deriv), deriv

    if method == "implicit":
        next_state = state.copy()
        for _ in range(max(1, int(implicit_iters))):
            next_state = state + dt * DYNAMICS_MODEL.dynamics(next_state, action)
        return next_state, deriv

    raise ValueError(f"Unsupported integration method: {method}")


def _load_dataset_trajectory(dataset_path: Path, trajectory: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(dataset_path) as data:
        if trajectory < 0 or trajectory >= int(data["dx"].shape[0]):
            raise IndexError(f"Trajectory {trajectory} is outside dataset range 0..{data['dx'].shape[0] - 1}.")

        length = int(data["dx"].shape[1])
        reference_states = np.zeros((length, len(STATE_LABELS)), dtype=np.float64)
        for idx, key in enumerate(STATE_DATASET_KEYS):
            reference_states[:, idx] = np.asarray(data[key][trajectory], dtype=np.float64)
        for offset, key in enumerate(EXT_MOMENT_KEYS, start=12):
            reference_states[:, offset] = float(np.asarray(data[key])[trajectory])
        reference_states[:, 15:19] = np.asarray(data["omega"][trajectory], dtype=np.float64)

        actions = np.asarray(data["u"][trajectory], dtype=np.float64)
        time = np.asarray(data["t"][trajectory], dtype=np.float64)
        if time.shape[0] >= 2:
            dt_steps = np.diff(time)
        else:
            # Hover datasets store dt at twice the sample spacing used by the simulators.
            dt_steps = np.asarray([0.5 * float(np.asarray(data["dt"])[trajectory])], dtype=np.float64)

    return reference_states, actions, time, dt_steps


def replay_commands(
    dataset_path: Path,
    trajectory: int,
    horizon: float,
    integration_method: str,
    implicit_iters: int,
    fixed_dt: float | None = None,
) -> dict[str, np.ndarray | float | int]:
    reference_states, dataset_actions, dataset_time, dataset_dt_steps = _load_dataset_trajectory(dataset_path, trajectory)
    if dataset_dt_steps.size == 0:
        raise ValueError("Dataset trajectory must contain at least one timestep.")
    nominal_dt = float(fixed_dt) if fixed_dt is not None else float(np.median(dataset_dt_steps))

    state = reference_states[0].copy()
    states = [state.copy()]
    applied_actions: list[np.ndarray] = []
    action_indices: list[int] = []
    action_times: list[float] = []
    dt_steps: list[float] = []
    repeated_last: list[bool] = []
    prev_deriv = None
    current_time = 0.0
    step_idx = 0

    while current_time < horizon - 1.0e-12:
        if fixed_dt is not None:
            dt = float(fixed_dt)
            is_repeated = step_idx >= dataset_actions.shape[0]
        elif step_idx < dataset_dt_steps.shape[0]:
            dt = float(dataset_dt_steps[step_idx])
            is_repeated = False
        else:
            dt = nominal_dt
            is_repeated = True
        dt = min(dt, horizon - current_time)

        # There are N state/action samples but only N-1 dataset intervals.
        # After the final timestamp, keep repeating the last recorded command.
        action_idx = min(step_idx, dataset_actions.shape[0] - 1)
        action = np.clip(dataset_actions[action_idx], 0.0, 1.0)
        applied_actions.append(action.copy())
        action_indices.append(action_idx)
        action_times.append(current_time)
        dt_steps.append(dt)
        repeated_last.append(is_repeated)
        state, prev_deriv = _integrate_state(
            integration_method,
            state,
            action,
            dt,
            prev_deriv=prev_deriv,
            implicit_iters=implicit_iters,
        )
        current_time += dt
        states.append(state.copy())
        step_idx += 1

    sim_time = np.concatenate([[0.0], np.cumsum(np.asarray(dt_steps, dtype=np.float64))])
    action_time = np.asarray(action_times, dtype=np.float64)
    states_array = np.asarray(states, dtype=np.float64)
    display_action_indices = np.minimum(np.arange(states_array.shape[0], dtype=int), dataset_actions.shape[0] - 1)
    display_actions = np.clip(dataset_actions[display_action_indices], 0.0, 1.0)
    return {
        "reference_states": reference_states,
        "reference_world": _body_to_world_trajectory(reference_states),
        "dataset_actions": dataset_actions,
        "dataset_time": dataset_time,
        "dataset_dt_steps": dataset_dt_steps,
        "states": states_array,
        "states_world": _body_to_world_trajectory(states_array),
        "actions": np.asarray(applied_actions, dtype=np.float64),
        "action_indices": np.asarray(action_indices, dtype=np.int64),
        "time": sim_time,
        "action_time": action_time,
        "display_actions": display_actions,
        "display_action_indices": display_action_indices,
        "dt_steps": np.asarray(dt_steps, dtype=np.float64),
        "repeated_last": np.asarray(repeated_last, dtype=bool),
        "dt": nominal_dt,
        "fixed_dt": np.nan if fixed_dt is None else float(fixed_dt),
        "trajectory": trajectory,
        "horizon": horizon,
    }


def replay_collocation_dataset(dataset_path: Path, trajectory: int) -> dict[str, np.ndarray | float | int]:
    """Replay the Hermite-Simpson dataset trajectory exactly as stored.

    The Bebop2 hover dataset has 2*n-1 samples: even samples are OCP nodes and
    odd samples are Hermite-Simpson midpoints. The midpoint controls are not
    zero-order-hold commands; they are the averaged controls used by the
    collocation constraints. This mode therefore animates the dataset-consistent
    state/control schedule directly instead of integrating with sample-and-hold.
    """
    reference_states, dataset_actions, dataset_time, dataset_dt_steps = _load_dataset_trajectory(dataset_path, trajectory)
    if dataset_dt_steps.size == 0:
        raise ValueError("Dataset trajectory must contain at least one timestep.")
    states_array = reference_states.copy()
    dt_steps = np.diff(dataset_time)
    actions = np.clip(dataset_actions[:-1], 0.0, 1.0)
    action_indices = np.arange(actions.shape[0], dtype=np.int64)
    display_action_indices = np.arange(dataset_actions.shape[0], dtype=np.int64)
    return {
        "reference_states": reference_states,
        "reference_world": _body_to_world_trajectory(reference_states),
        "dataset_actions": dataset_actions,
        "dataset_time": dataset_time,
        "dataset_dt_steps": dataset_dt_steps,
        "states": states_array,
        "states_world": _body_to_world_trajectory(states_array),
        "actions": actions,
        "action_indices": action_indices,
        "time": dataset_time.copy(),
        "action_time": dataset_time[:-1].copy(),
        "display_actions": np.clip(dataset_actions, 0.0, 1.0),
        "display_action_indices": display_action_indices,
        "dt_steps": dt_steps,
        "repeated_last": np.zeros(actions.shape[0], dtype=bool),
        "dt": float(np.median(dataset_dt_steps)),
        "fixed_dt": np.nan,
        "trajectory": trajectory,
        "horizon": float(dataset_time[-1]),
    }


def replay_collocation_then_hold(
    dataset_path: Path,
    trajectory: int,
    horizon: float,
    integration_method: str,
    implicit_iters: int,
    fixed_dt: float | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Follow stored collocation points, then hold the final command."""
    reference_states, dataset_actions, dataset_time, dataset_dt_steps = _load_dataset_trajectory(dataset_path, trajectory)
    if dataset_dt_steps.size == 0:
        raise ValueError("Dataset trajectory must contain at least one timestep.")

    nominal_dt = float(fixed_dt) if fixed_dt is not None else float(np.median(dataset_dt_steps))
    dataset_end = float(dataset_time[-1])
    horizon = max(float(horizon), dataset_end)

    states: list[np.ndarray] = [state.copy() for state in reference_states]
    actions: list[np.ndarray] = [action.copy() for action in np.clip(dataset_actions[:-1], 0.0, 1.0)]
    action_indices: list[int] = list(range(dataset_actions.shape[0] - 1))
    action_times: list[float] = [float(value) for value in dataset_time[:-1]]
    dt_steps: list[float] = [float(value) for value in np.diff(dataset_time)]
    repeated_last: list[bool] = [False] * (dataset_actions.shape[0] - 1)

    state = reference_states[-1].copy()
    last_action = np.clip(dataset_actions[-1], 0.0, 1.0)
    prev_deriv = None
    current_time = dataset_end

    while current_time < horizon - 1.0e-12:
        dt = min(nominal_dt, horizon - current_time)
        actions.append(last_action.copy())
        action_indices.append(dataset_actions.shape[0] - 1)
        action_times.append(current_time)
        dt_steps.append(dt)
        repeated_last.append(True)
        state, prev_deriv = _integrate_state(
            integration_method,
            state,
            last_action,
            dt,
            prev_deriv=prev_deriv,
            implicit_iters=implicit_iters,
        )
        current_time += dt
        states.append(state.copy())

    states_array = np.asarray(states, dtype=np.float64)
    sim_time = np.concatenate([[0.0], np.cumsum(np.asarray(dt_steps, dtype=np.float64))])
    display_action_indices = np.minimum(np.arange(states_array.shape[0], dtype=int), dataset_actions.shape[0] - 1)
    display_actions = np.clip(dataset_actions[display_action_indices], 0.0, 1.0)
    if states_array.shape[0] > dataset_actions.shape[0]:
        display_actions[dataset_actions.shape[0] :] = last_action
        display_action_indices[dataset_actions.shape[0] :] = dataset_actions.shape[0] - 1

    return {
        "reference_states": reference_states,
        "reference_world": _body_to_world_trajectory(reference_states),
        "dataset_actions": dataset_actions,
        "dataset_time": dataset_time,
        "dataset_dt_steps": dataset_dt_steps,
        "states": states_array,
        "states_world": _body_to_world_trajectory(states_array),
        "actions": np.asarray(actions, dtype=np.float64),
        "action_indices": np.asarray(action_indices, dtype=np.int64),
        "time": sim_time,
        "action_time": np.asarray(action_times, dtype=np.float64),
        "display_actions": display_actions,
        "display_action_indices": display_action_indices,
        "dt_steps": np.asarray(dt_steps, dtype=np.float64),
        "repeated_last": np.asarray(repeated_last, dtype=bool),
        "dt": nominal_dt,
        "fixed_dt": np.nan if fixed_dt is None else float(fixed_dt),
        "trajectory": trajectory,
        "horizon": horizon,
    }


def _rpm(actions: np.ndarray) -> np.ndarray:
    info = DYNAMICS_MODEL.INFO
    return info.omega_min + np.clip(actions, 0.0, 1.0) * (info.omega_max - info.omega_min)


def _save_npz(path: Path, result: dict[str, np.ndarray | float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{key: value for key, value in result.items() if isinstance(value, np.ndarray)},
        dt=float(result["dt"]),
        fixed_dt=float(result["fixed_dt"]),
        trajectory=int(result["trajectory"]),
        horizon=float(result["horizon"]),
        dynamics_model=DYNAMICS_MODEL.INFO.name,
        tau=DYNAMICS_MODEL.INFO.tau,
    )


def _save_csv(path: Path, result: dict[str, np.ndarray | float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    states = np.asarray(result["states"], dtype=np.float64)
    actions = np.asarray(result["actions"], dtype=np.float64)
    time = np.asarray(result["time"], dtype=np.float64)
    dt_steps = np.asarray(result["dt_steps"], dtype=np.float64)
    repeated_last = np.asarray(result["repeated_last"], dtype=bool)
    header = ["t", *STATE_LABELS, "dt_step", "u1", "u2", "u3", "u4", "rpm1_cmd", "rpm2_cmd", "rpm3_cmd", "rpm4_cmd", "repeated_last"]
    rpm = _rpm(actions)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for idx, state in enumerate(states):
            if idx < actions.shape[0]:
                dt_value = dt_steps[idx]
                action_values = actions[idx]
                rpm_values = rpm[idx]
                repeated = int(repeated_last[idx])
            else:
                dt_value = np.nan
                action_values = [np.nan] * 4
                rpm_values = [np.nan] * 4
                repeated = ""
            writer.writerow([time[idx], *state, dt_value, *action_values, *rpm_values, repeated])


def _plot_trajectory(path: Path, result: dict[str, np.ndarray | float | int]) -> None:
    simulated = np.asarray(result["states_world"], dtype=np.float64)
    reference = np.asarray(result["reference_world"], dtype=np.float64)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(reference[:, 0], reference[:, 1], reference[:, 2], label="dataset reference", linewidth=1.8)
    ax.plot(simulated[:, 0], simulated[:, 1], simulated[:, 2], label="replayed commands", linewidth=1.8)
    ax.scatter(simulated[0, 0], simulated[0, 1], simulated[0, 2], s=35, label="start")
    ax.scatter(simulated[-1, 0], simulated[-1, 1], simulated[-1, 2], s=35, label="final")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"Trajectory {int(result['trajectory'])} replay: dataset commands through {DYNAMICS_MODEL.INFO.name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_states(path: Path, result: dict[str, np.ndarray | float | int]) -> None:
    time = np.asarray(result["time"], dtype=np.float64)
    states = np.asarray(result["states"], dtype=np.float64)
    groups = [
        ("position error [m]", ["dx", "dy", "dz"], slice(0, 3)),
        ("velocity [m/s]", ["vx", "vy", "vz"], slice(3, 6)),
        ("attitude [rad]", ["phi", "theta", "psi"], slice(6, 9)),
        ("body rates [rad/s]", ["p", "q", "r"], slice(9, 12)),
        ("motor speed [RPM]", ["omega1", "omega2", "omega3", "omega4"], slice(15, 19)),
    ]
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 13), sharex=True)
    for ax, (ylabel, labels, indices) in zip(axes, groups, strict=True):
        for label, column in zip(labels, range(indices.start, indices.stop), strict=True):
            ax.plot(time, states[:, column], label=label, linewidth=1.2)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("State response with dataset commands, holding last command after dataset horizon")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_actions(path: Path, result: dict[str, np.ndarray | float | int]) -> None:
    action_time = np.asarray(result["action_time"], dtype=np.float64)
    actions = np.asarray(result["actions"], dtype=np.float64)
    dataset_actions = np.asarray(result["dataset_actions"], dtype=np.float64)
    dataset_time = np.asarray(result["dataset_time"], dtype=np.float64)
    repeated_start = dataset_time[-1] if len(dataset_time) else 0.0
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for motor_idx, ax in enumerate(axes):
        ax.plot(action_time, actions[:, motor_idx], label=f"applied u{motor_idx + 1}", linewidth=1.4)
        ax.plot(dataset_time[: dataset_actions.shape[0]], dataset_actions[:, motor_idx], "--", label="dataset", linewidth=1.0)
        ax.axvline(repeated_start, color="k", linestyle=":", linewidth=1.0, label="hold last" if motor_idx == 0 else None)
        ax.set_ylabel(f"u{motor_idx + 1}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Motor commands replayed from dataset")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    default_dataset = PROJECT_ROOT / "datasets" / "hover_dataset_test_bebop2_corrected.npz"
    default_output_dir = PROJECT_ROOT / "organized_plots" / "sl_runs" / "dataset_command_replay"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--trajectory", type=int, default=88)
    parser.add_argument("--dynamics-model", default="quadrotor_sim_matlab", choices=available_dynamics_models())
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--stop-at-dataset-end", action="store_true", help="Use the dataset final timestamp as the simulation horizon.")
    parser.add_argument("--collocation-playback", action="store_true", help="Use the stored Hermite-Simpson states/controls exactly as the dataset defines them.")
    parser.add_argument("--collocation-then-hold", action="store_true", help="Use stored Hermite-Simpson points until dataset end, then hold the final command.")
    parser.add_argument("--integration-method", default="rk4", choices=["explicit", "euler", "rk4", "adams-bashforth", "adams-moulton", "implicit"])
    parser.add_argument("--implicit-iters", type=int, default=5)
    parser.add_argument("--fixed-dt", type=float, default=None, help="Override dataset time steps, e.g. 0.01 to reproduce legacy fixed-dt videos.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    return parser.parse_args(cli_args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)
    _set_dynamics_model(args.dynamics_model)
    horizon = float(args.horizon)
    if args.collocation_then_hold:
        result = replay_collocation_then_hold(
            dataset_path=args.dataset,
            trajectory=args.trajectory,
            horizon=horizon,
            integration_method=args.integration_method,
            implicit_iters=args.implicit_iters,
            fixed_dt=args.fixed_dt,
        )
    elif args.collocation_playback:
        result = replay_collocation_dataset(args.dataset, args.trajectory)
    else:
        if args.stop_at_dataset_end:
            _, _, dataset_time, _ = _load_dataset_trajectory(args.dataset, args.trajectory)
            horizon = float(dataset_time[-1])

        result = replay_commands(
            dataset_path=args.dataset,
            trajectory=args.trajectory,
            horizon=horizon,
            integration_method=args.integration_method,
            implicit_iters=args.implicit_iters,
            fixed_dt=args.fixed_dt,
        )
    dt_tag = "" if args.fixed_dt is None else f"_fixeddt{str(args.fixed_dt).replace('.', 'p')}"
    if args.collocation_then_hold:
        horizon_tag = f"collocation_then_hold_{str(args.horizon).replace('.', 'p')}s"
    elif args.collocation_playback:
        horizon_tag = "collocation_dataset"
    else:
        horizon_tag = "datasetend" if args.stop_at_dataset_end else f"{str(args.horizon).replace('.', 'p')}s"
    tau_tag = str(DYNAMICS_MODEL.INFO.tau).replace(".", "p")
    stem = f"{args.dataset.stem}_traj{args.trajectory:03d}_{DYNAMICS_MODEL.INFO.name}_tau{tau_tag}_replay_{horizon_tag}{dt_tag}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / f"{stem}.npz"
    csv_path = args.output_dir / f"{stem}.csv"
    trajectory_plot = args.output_dir / f"{stem}_trajectory.png"
    states_plot = args.output_dir / f"{stem}_states.png"
    actions_plot = args.output_dir / f"{stem}_actions.png"

    _save_npz(npz_path, result)
    _save_csv(csv_path, result)
    _plot_trajectory(trajectory_plot, result)
    _plot_states(states_plot, result)
    _plot_actions(actions_plot, result)

    states = np.asarray(result["states"], dtype=np.float64)
    actions = np.asarray(result["actions"], dtype=np.float64)
    repeated_last = np.asarray(result["repeated_last"], dtype=bool)
    print(f"dataset: {args.dataset}")
    print(f"trajectory: {args.trajectory}")
    print(f"dynamics: {DYNAMICS_MODEL.INFO.name}, tau={DYNAMICS_MODEL.INFO.tau:.3f}s")
    print(f"dt: {float(result['dt']):.9f}s")
    print(f"simulated states: {states.shape[0]} ({float(result['time'][-1]):.6f}s)")
    print(f"applied commands: {actions.shape[0]}, repeated-last steps: {int(repeated_last.sum())}")
    print(f"last dataset command: {np.asarray(result['dataset_actions'])[-1]}")
    print(f"final body state [{', '.join(STATE_LABELS[:12])}]:")
    print(np.array2string(states[-1, :12], precision=6, suppress_small=False))
    print(f"final motor rpm: {np.array2string(states[-1, 15:19], precision=3)}")
    print(f"wrote: {npz_path}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {trajectory_plot}")
    print(f"wrote: {states_plot}")
    print(f"wrote: {actions_plot}")


if __name__ == "__main__":
    main()
