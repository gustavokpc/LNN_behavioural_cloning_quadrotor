#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Race-gate simulation using the exported C controllers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from utils.c_controller import CController, c_dir_from_model_path
from utils.config import load_yaml, resolve_saved_config
from utils.normalization_limits import OMEGA_MAX, OMEGA_MIN
from utils.quadrotor_sim import body_to_world_trajectory, world_to_body_state
from utils.quadrotor_sim_c import rollout_c_controller


def _wrap_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


def _default_race_map() -> dict:
    radius = 1.5
    gate_pos = np.array(
        [
            [radius, -radius, -1.5],
            [0.0, 0.0, -1.5],
            [-radius, radius, -1.5],
            [0.0, 2.0 * radius, -1.5],
            [radius, radius, -1.5],
            [0.0, 0.0, -1.5],
            [-radius, -radius, -1.5],
            [0.0, -2.0 * radius, -1.5],
        ],
        dtype=np.float64,
    )
    gate_yaw = np.array([1, 2, 1, 0, -1, -2, -1, 0], dtype=np.float64) * np.pi / 2.0
    return {"targets": gate_pos, "psi_ref": gate_yaw}


def simulate_race_c(config_sim: dict, config_model: dict, project_root: Path, c_model_dir: Path | None = None):
    c_dir = c_model_dir or c_dir_from_model_path(config_sim["model_path"], project_root)
    controller = CController(c_dir)
    sim_cfg = config_sim["simulation"]
    dt = float(sim_cfg["dt"])
    max_total_steps = int(round(sim_cfg["time_simulation"] / dt))
    race_map = _default_race_map()

    current_world = np.zeros(19, dtype=np.float64)
    current_world[15:19] = (OMEGA_MAX + OMEGA_MIN) / 2.0
    current_world[0:3] = race_map["targets"][0]

    total_states = []
    total_actions = []
    gates_passed = 0
    elapsed_steps = 0
    gate_index = 1

    while elapsed_steps < max_total_steps:
        target = race_map["targets"][gate_index]
        psi_ref = race_map["psi_ref"][gate_index - 1]

        relative_world = current_world.copy()
        relative_world[0:3] -= target
        relative_world[8] = _wrap_angle(relative_world[8] - psi_ref)
        initial_body = world_to_body_state(relative_world)

        states_body, actions = rollout_c_controller(
            controller=controller,
            initial_state=initial_body,
            input_labels=config_model["dataset"]["input_labels"],
            dt=dt,
            horizon_steps=max_total_steps - elapsed_steps,
            integration_method=sim_cfg.get("integration_method", "explicit"),
            implicit_iters=int(sim_cfg.get("implicit_iterations", 5)),
            stop_fn=lambda state, step: np.linalg.norm(state[0:3]) < sim_cfg["dist_error"],
            reset_controller=True,
        )

        states_world = body_to_world_trajectory(states_body)
        states_world[:, 0:3] += target
        states_world[:, 8] += psi_ref

        if total_states:
            total_states.append(states_world[1:])
        else:
            total_states.append(states_world)
        if actions.size:
            total_actions.append(actions)

        current_world = states_world[-1].copy()
        elapsed_steps += max(len(states_world) - 1, 0)
        gates_passed += 1
        gate_index = (gate_index + 1) % len(race_map["targets"])

    return np.vstack(total_states), (np.vstack(total_actions) if total_actions else np.zeros((0, 4))), gates_passed


def _plot_race(states_world: np.ndarray, actions: np.ndarray) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(states_world[:, 0], states_world[:, 1], states_world[:, 2], linewidth=2.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Race Drone Trajectory - C Controller")
    fig.tight_layout()

    if actions.size:
        _, axes = plt.subplots(2, 2, figsize=(15, 5))
        axes = axes.flatten()
        for idx in range(4):
            axes[idx].plot(actions[:, idx])
            axes[idx].set_title(f"u_{idx + 1}")
            axes[idx].grid(True)
        plt.tight_layout()
    plt.show()


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Race-gate simulation with C-exported controllers.")
    default_root = Path(__file__).resolve().parent
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

    states_world, actions, gates_passed = simulate_race_c(config_sim, config_model, args.project_root, args.c_model_dir)
    total_energy = float(actions.sum() * config_sim["simulation"]["dt"]) if actions.size else 0.0
    total_time = float(max(len(states_world) - 1, 0) * config_sim["simulation"]["dt"])
    print(f"Gates passed: {gates_passed}")
    print(f"Total time: {total_time:.6f}s")
    print(f"Total energy: {total_energy:.6f}")
    if gates_passed:
        print(f"Average time per gate: {total_time / gates_passed:.6f}s")
        print(f"Average energy per gate: {total_energy / gates_passed:.6f}")

    if args.plot or config_sim.get("show_sim", False):
        _plot_race(states_world, actions)


if __name__ == "__main__":
    main()
