#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset-initialized rollouts using the exported C controllers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from utils.c_controller import CController, c_dir_from_model_path
from utils.config import load_yaml, resolve_saved_config
from utils.data import expand_feature_labels, get_data
from utils.quadrotor_sim import body_to_world_trajectory, input_labels_without_time, state_from_input_features
from utils.quadrotor_sim_c import rollout_c_controller


def _first_target_step(states_body: np.ndarray, dist_error: float, vel_error: float, ang_error: float) -> int | None:
    for idx, state in enumerate(states_body):
        if (
            np.linalg.norm(state[0:3]) < dist_error
            and np.linalg.norm(state[3:6]) < vel_error
            and np.linalg.norm(state[6:8]) < ang_error
        ):
            return idx
    return None


def _prepare_dataset(config_model: dict, dataset_path: str):
    model_input_labels = config_model["dataset"]["input_labels"]
    dataset_input_labels = input_labels_without_time(model_input_labels) + ["dt"]

    inputs, outputs = get_data(
        input_labels=dataset_input_labels,
        output_labels=config_model["dataset"]["output_labels"],
        path=dataset_path,
        normalized=False,
    )

    expanded_labels = expand_feature_labels(dataset_input_labels)
    dt_index = expanded_labels.index("dt")
    dt_channel = inputs[:, dt_index, :]
    if not np.all(np.isfinite(dt_channel)):
        raise ValueError("Dataset dt channel contains non-finite values.")
    if not np.allclose(dt_channel, dt_channel[:, :1]):
        raise ValueError("Dataset dt must be constant within each trajectory.")

    dt_values = dt_channel[:, 0]
    keep_indices = [idx for idx, label in enumerate(expanded_labels) if label != "dt"]
    inputs = inputs[:, keep_indices, :]
    return inputs, outputs, dt_values


def simulate_from_dataset_c(config_sim: dict, config_model: dict, project_root: Path, c_model_dir: Path | None = None):
    c_dir = c_model_dir or c_dir_from_model_path(config_sim["model_path"], project_root)
    controller = CController(c_dir)
    raw_inputs, raw_outputs, dt_values = _prepare_dataset(config_model, config_sim["dataset"]["path_sim"])
    base_labels = input_labels_without_time(config_model["dataset"]["input_labels"])
    expanded_labels = expand_feature_labels(base_labels)
    sim_cfg = config_sim["simulation"]

    results = []
    first_plot_payload = None

    for traj_idx in range(raw_inputs.shape[0]):
        dt = float(dt_values[traj_idx])
        raw_traj_inputs = raw_inputs[traj_idx]
        initial_state = state_from_input_features(raw_traj_inputs[:, 0], expanded_labels)
        reference_states = np.asarray(
            [state_from_input_features(raw_traj_inputs[:, step], expanded_labels) for step in range(raw_traj_inputs.shape[1])],
            dtype=np.float64,
        )
        horizon_cap = max(1, int(round(sim_cfg["time_simulation"] / dt)))
        simulated_states_body, generated_actions = rollout_c_controller(
            controller=controller,
            initial_state=initial_state,
            input_labels=config_model["dataset"]["input_labels"],
            dt=dt,
            horizon_steps=horizon_cap,
            integration_method=sim_cfg.get("integration_method", "explicit"),
            implicit_iters=int(sim_cfg.get("implicit_iterations", 5)),
            stop_fn=None,
        )

        simulated_target_step = _first_target_step(
            simulated_states_body,
            sim_cfg["dist_error"],
            sim_cfg["vel_error"],
            sim_cfg["ang_error"],
        )
        reference_target_step = _first_target_step(
            reference_states,
            sim_cfg["dist_error"],
            sim_cfg["vel_error"],
            sim_cfg["ang_error"],
        )

        if simulated_target_step is None:
            print(f"Trajectory {traj_idx}: simulated trajectory did not reach target within horizon.")
        else:
            sim_energy = float(generated_actions[:max(simulated_target_step, 1)].sum() * dt)
            ref_energy = float(raw_outputs[traj_idx, :, :max(reference_target_step, 1)].sum() * dt)
            results.append(
                {
                    "trajectory": traj_idx,
                    "sim_energy": sim_energy,
                    "ref_energy": ref_energy,
                    "sim_target_step": simulated_target_step,
                    "ref_target_step": reference_target_step,
                    "failed": simulated_target_step is None,
                }
            )

        if first_plot_payload is None:
            first_plot_payload = {
                "reference_world": body_to_world_trajectory(reference_states[:horizon_cap]),
                "simulated_world": body_to_world_trajectory(simulated_states_body),
                "reference_actions": raw_outputs[traj_idx, :, :horizon_cap].transpose(1, 0),
                "simulated_actions": generated_actions,
            }

    return results, first_plot_payload


def _plot_payload(payload: dict) -> None:
    reference_world = payload["reference_world"]
    simulated_world = payload["simulated_world"]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(reference_world[:, 0], reference_world[:, 1], reference_world[:, 2], label="Reference")
    ax.plot(simulated_world[:, 0], simulated_world[:, 1], simulated_world[:, 2], label="Simulated C")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Dataset-Initialized Rollout - C Controller")
    ax.legend()
    fig.tight_layout()

    ref_actions = payload["reference_actions"]
    sim_actions = payload["simulated_actions"]
    _, axes = plt.subplots(2, 2, figsize=(15, 5))
    axes = axes.flatten()
    for idx in range(4):
        axes[idx].plot(ref_actions[:, idx], label="Reference")
        axes[idx].plot(sim_actions[:, idx], label="Simulated C")
        axes[idx].set_title(f"u_{idx + 1}")
        axes[idx].grid(True)
        axes[idx].legend()
    plt.tight_layout()
    plt.show()


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset-start simulations with C-exported controllers.")
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
    results, payload = simulate_from_dataset_c(config_sim, config_model, args.project_root, args.c_model_dir)

    energy_delta = np.array([item["sim_energy"] - item["ref_energy"] for item in results], dtype=np.float64)
    failed_runs = sum(1 for item in results if item["failed"])
    if energy_delta.size:
        print(f"Mean energy delta: {energy_delta.mean():.6f} ± {energy_delta.std():.6f}")
    else:
        print("Mean energy delta: no successful runs")
    print(f"Failed runs: {failed_runs}/{len(results)}")

    if args.plot or config_sim.get("show_sim", False):
        _plot_payload(payload)


if __name__ == "__main__":
    main()
