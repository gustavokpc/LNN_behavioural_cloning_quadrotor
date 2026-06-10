#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random-start simulations using the exported C neural-network controllers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from utils.c_controller import CController, c_dir_from_model_path
from utils.config import load_yaml, resolve_saved_config
from utils.quadrotor_sim import body_to_world_trajectory, generate_starting_conditions
from utils.quadrotor_sim_c import rollout_c_controller


def _reached_target(state: np.ndarray, dist_error: float, vel_error: float, ang_error: float) -> bool:
    return (
        np.linalg.norm(state[0:3]) < dist_error
        and np.linalg.norm(state[3:6]) < vel_error
        and np.linalg.norm(state[6:8]) < ang_error
    )


def simulate_random_starts_c(config_sim: dict, config_model: dict, project_root: Path, c_model_dir: Path | None = None):
    c_dir = c_model_dir or c_dir_from_model_path(config_sim["model_path"], project_root)
    controller = CController(c_dir)
    sim_cfg = config_sim["simulation"]
    dt = float(sim_cfg["dt"])
    horizon_steps = int(round(sim_cfg["time_simulation"] / dt))

    starts = generate_starting_conditions(int(config_sim["number_simulations"]), seed=int(config_sim.get("seed", 42)))
    trajectories_world = []
    actions_all = []
    energies = []
    times_to_target = []
    failed_runs = 0

    for start_state in starts:
        states_body, actions = rollout_c_controller(
            controller=controller,
            initial_state=start_state,
            input_labels=config_model["dataset"]["input_labels"],
            dt=dt,
            horizon_steps=horizon_steps,
            integration_method=sim_cfg.get("integration_method", "explicit"),
            implicit_iters=int(sim_cfg.get("implicit_iterations", 5)),
            stop_fn=lambda state, step: _reached_target(
                state,
                sim_cfg["dist_error"],
                sim_cfg["vel_error"],
                sim_cfg["ang_error"],
            ),
        )
        trajectories_world.append(body_to_world_trajectory(states_body))
        actions_all.append(actions)
        energies.append(float(actions.sum() * dt) if actions.size else 0.0)

        if _reached_target(states_body[-1], sim_cfg["dist_error"], sim_cfg["vel_error"], sim_cfg["ang_error"]):
            times_to_target.append(float(max(len(states_body) - 1, 0) * dt))
        else:
            failed_runs += 1

    return starts, trajectories_world, actions_all, np.asarray(energies), np.asarray(times_to_target), failed_runs


def _plot_trajectories(trajectories_world: list[np.ndarray], actions_all: list[np.ndarray]) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for idx, trajectory in enumerate(trajectories_world[: min(20, len(trajectories_world))]):
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], alpha=0.7, label=f"Traj {idx + 1}")
    ax.scatter(0.0, 0.0, 0.0, c="red", s=40, label="Target")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Random Start Rollouts - C Controller")
    fig.tight_layout()

    if actions_all:
        commands = actions_all[0]
        _, axes = plt.subplots(2, 2, figsize=(15, 5))
        axes = axes.flatten()
        for idx in range(4):
            axes[idx].plot(commands[:, idx])
            axes[idx].set_title(f"u_{idx + 1}")
            axes[idx].grid(True)
        plt.tight_layout()
    plt.show()


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-start simulations with C-exported controllers.")
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=default_root / "simulator_config.yaml")
    parser.add_argument("--config-dir", type=Path, default=default_root / "configs")
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--c-model-dir", type=Path, default=None)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args(cli_args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)
    config_sim = load_yaml(args.config)
    config_model = load_yaml(resolve_saved_config(config_sim["model_path"], args.config_dir))
    starts, trajectories_world, actions_all, energies, times_to_target, failed_runs = simulate_random_starts_c(
        config_sim,
        config_model,
        args.project_root,
        args.c_model_dir,
    )
    print(f"Average energy: {energies.mean():.6f} ± {energies.std():.6f}")
    if times_to_target.size:
        print(f"Average time to target: {times_to_target.mean():.6f}s ± {times_to_target.std():.6f}s")
    print(f"Failed runs: {failed_runs}/{starts.shape[0]}")

    if args.plot or config_sim.get("show_sim", False):
        _plot_trajectories(trajectories_world, actions_all)


if __name__ == "__main__":
    main()
