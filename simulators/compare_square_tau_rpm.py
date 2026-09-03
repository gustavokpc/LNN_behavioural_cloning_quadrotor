#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare square-rollout RPM variation across motor time constants."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from .Simulator_gazebo_square_C import (
    _network_world_to_enu,
    resolve_model_paths,
    simulate_gazebo_square,
)
from ..utils.config import load_yaml
from ..utils.dynamics_models import get_dynamics_info


def _rpm_metrics(rpm: np.ndarray, prefix: str) -> dict[str, float]:
    if rpm.size == 0:
        return {
            f"{prefix}_std_rpm": float("nan"),
            f"{prefix}_mean_abs_delta_rpm": float("nan"),
            f"{prefix}_rms_delta_rpm": float("nan"),
            f"{prefix}_max_abs_delta_rpm": float("nan"),
        }
    delta = np.diff(rpm, axis=0)
    return {
        f"{prefix}_std_rpm": float(np.std(rpm)),
        f"{prefix}_mean_abs_delta_rpm": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        f"{prefix}_rms_delta_rpm": float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
        f"{prefix}_max_abs_delta_rpm": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def compare(args: argparse.Namespace) -> list[dict[str, float | int]]:
    model_config, c_model_dir = resolve_model_paths(args.project_root, args.model, args.model_config, args.c_model_dir)
    config_model = load_yaml(model_config)
    rows: list[dict[str, float | int]] = []

    for tau in args.tau:
        states_world, actions, _, gates_passed = simulate_gazebo_square(
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
            tau=tau,
        )

        steps = min(len(states_world), len(actions))
        states = states_world[:steps]
        actions01 = np.asarray(actions[:steps], dtype=np.float64)
        info = get_dynamics_info()
        rpm_ref = info.omega_min + np.clip(actions01, 0.0, 1.0) * (info.omega_max - info.omega_min)
        rpm_obs = states[:, 15:19]
        final_enu = _network_world_to_enu(states_world[-1, 0:3])

        row: dict[str, float | int] = {
            "tau": float(tau),
            "steps": int(max(len(states_world) - 1, 0)),
            "gates_passed": int(gates_passed),
            "final_x_enu": float(final_enu[0]),
            "final_y_enu": float(final_enu[1]),
            "final_z_enu": float(final_enu[2]),
        }
        row.update(_rpm_metrics(rpm_ref, "rpm_ref"))
        row.update(_rpm_metrics(rpm_obs, "rpm_obs"))
        rows.append(row)

    return rows


def _write_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, float | int]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tau = np.asarray([row["tau"] for row in rows], dtype=np.float64)
    metrics = [
        ("rpm_ref_mean_abs_delta_rpm", "rpm_ref mean |delta| [RPM/step]"),
        ("rpm_obs_mean_abs_delta_rpm", "rpm_obs mean |delta| [RPM/step]"),
        ("rpm_ref_std_rpm", "rpm_ref std [RPM]"),
        ("rpm_obs_std_rpm", "rpm_obs std [RPM]"),
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 10), sharex=True)
    for ax, (key, label) in zip(axes, metrics, strict=True):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        ax.plot(tau, values, marker="o")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("tau [s]")
    fig.suptitle("Square rollout RPM variation vs tau")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    runs_dir = project_root / "organized_plots" / "sl_runs" / "comparisons"
    parser = argparse.ArgumentParser(description="Compare square-rollout RPM variation across tau values.")
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--model", default="CFC")
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--c-model-dir", type=Path, default=None)
    parser.add_argument("--dynamics-model", default="quadrotor_sim_matlab")
    parser.add_argument("--tau", type=float, nargs="+", default=[0.02, 0.04, 0.06])
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--time-simulation", type=float, default=20.0)
    parser.add_argument("--dist-error", type=float, default=0.001)
    parser.add_argument("--integration-method", default="rk4")
    parser.add_argument("--implicit-iters", type=int, default=1)
    parser.add_argument("--start-waypoint-index", type=int, default=3)
    parser.add_argument("--start-alt", type=float, default=1.0)
    parser.add_argument("--waypoint-alt", type=float, default=1.5)
    parser.add_argument("--reset-each-waypoint", action="store_true")
    parser.add_argument("--csv-output", type=Path, default=runs_dir / "square_tau_rpm_metrics.csv")
    parser.add_argument("--plot-output", type=Path, default=runs_dir / "square_tau_rpm_metrics.png")
    return parser.parse_args(cli_args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)
    rows = compare(args)
    _write_csv(rows, args.csv_output)
    _plot(rows, args.plot_output)
    print(f"Metrics CSV saved to {args.csv_output}")
    print(f"Metrics plot saved to {args.plot_output}")
    for row in rows:
        print(
            f"tau={row['tau']:.3f} | "
            f"ref mean|dRPM|={row['rpm_ref_mean_abs_delta_rpm']:.3f} | "
            f"obs mean|dRPM|={row['rpm_obs_mean_abs_delta_rpm']:.3f} | "
            f"ref std={row['rpm_ref_std_rpm']:.3f} | "
            f"obs std={row['rpm_obs_std_rpm']:.3f}"
        )


if __name__ == "__main__":
    main()
