#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roll out the controller from dataset initial conditions and compare to reference trajectories.

This script answers the question: "if I initialize the simulator from the same
state found in the dataset, how does the closed-loop controller compare against
the recorded reference commands and state evolution?"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.config import load_yaml, resolve_saved_config
from utils.data import expand_feature_labels, get_data
from utils.quadrotor_sim import (
    body_to_world_trajectory,
    build_lightning_model,
    input_labels_without_time,
    normalize_input,
    rollout_controller,
    state_from_input_features,
)


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
        # desired_trajectories=20,
        path=dataset_path,
        normalized=False,
    )

    expanded_labels = expand_feature_labels(dataset_input_labels)
    dt_index = expanded_labels.index("dt")
    dt_channel = inputs[:, dt_index, :]
    if not np.all(np.isfinite(dt_channel)):
        raise ValueError("Dataset dt channel contains non-finite values.")

    # Closed-loop rollout currently accepts one scalar timestep per trajectory.
    # Validate that the dataset dt is constant within each trajectory before
    # extracting that scalar.
    if not np.allclose(dt_channel, dt_channel[:, :1]):
        raise ValueError(
            "Simulator_start_dataset expects a constant dataset dt within each trajectory."
        )
    dt_values = dt_channel[:, 0]

    # Remove the auxiliary dataset dt channel before reconstructing the
    # physical-state observations expected by the simulator/model window.
    keep_indices = [idx for idx, label in enumerate(expanded_labels) if label != "dt"]
    inputs = inputs[:, keep_indices, :]
    return inputs, outputs, dt_values


def _initial_window(raw_traj_inputs: np.ndarray, base_labels: list[str], seq_len: int) -> np.ndarray:
    feature_traj = raw_traj_inputs.transpose(1, 0)
    # When sequence models are used, the initial hidden context is taken from
    # the first observed states in the trajectory rather than repeated zeros.
    window = feature_traj[: max(1, seq_len)]
    if window.shape[0] < seq_len:
        pad = np.repeat(window[-1:, :], seq_len - window.shape[0], axis=0)
        window = np.concatenate([window, pad], axis=0)
    return np.asarray([normalize_input(row, base_labels) for row in window], dtype=np.float64)


def simulate_from_dataset(config_sim: dict, config_model: dict, project_root: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_lightning_model(config_model, config_sim["model_path"], project_root, device)

    raw_inputs, raw_outputs, dt_values = _prepare_dataset(config_model, config_sim["dataset"]["path_sim"])
    base_labels = input_labels_without_time(config_model["dataset"]["input_labels"])
    expanded_labels = expand_feature_labels(base_labels)
    use_sequencing = bool(config_model.get("sequencing", {}).get("value", False))
    seq_len = int(config_model.get("sequencing", {}).get("seq_len", 1))
    sim_cfg = config_sim["simulation"]

    results = []
    first_plot_payload = None

    for traj_idx in range(raw_inputs.shape[0]):
        dt = float(dt_values[traj_idx])
        raw_traj_inputs = raw_inputs[traj_idx]
        initial_state = state_from_input_features(raw_traj_inputs[:, 0], expanded_labels)
        init_window = _initial_window(raw_traj_inputs, base_labels, seq_len)
        # Reconstruct the dataset trajectory as a full body-frame state rollout
        # so we can compare apples-to-apples with the simulated trajectory.
        reference_states = np.asarray(
            [state_from_input_features(raw_traj_inputs[:, step], expanded_labels) for step in range(raw_traj_inputs.shape[1])],
            dtype=np.float64,
        )

        horizon_cap = max(1, int(round(sim_cfg["time_simulation"] / dt)))
        # horizon_steps = min(horizon_cap, raw_outputs.shape[2])
        simulated_states_body, generated_actions = rollout_controller(
            model=model,
            initial_state=initial_state,
            input_labels=config_model["dataset"]["input_labels"],
            dt=dt,
            horizon_steps=horizon_cap,
            device=device,
            use_sequencing=use_sequencing,
            seq_len=seq_len,
            initial_window=init_window,
            force_timespans=True,
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

        # sim_energy = float(generated_actions.sum() * dt) if generated_actions.size else 0.0
        if simulated_target_step is None:
            print(f"Trajectory {traj_idx}: simulated trajectory did not reach target within horizon.")
            # sim_energy = float(generated_actions[:horizon_cap].sum() * dt)
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

        # Keep only the first trajectory payload for optional plotting; the
        # aggregate metrics are computed over all trajectories above.
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
    ax.plot(simulated_world[:, 0], simulated_world[:, 1], simulated_world[:, 2], label="Simulated")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Dataset-Initialised Closed-Loop Rollout")
    ax.legend()
    fig.tight_layout()

    ref_actions = payload["reference_actions"]
    sim_actions = payload["simulated_actions"]
    _, axes = plt.subplots(2, 2, figsize=(15, 5))
    axes = axes.flatten()
    for idx in range(4):
        axes[idx].plot(ref_actions[:, idx], label="Reference")
        axes[idx].plot(sim_actions[:, idx], label="Simulated")
        axes[idx].set_title(f"u_{idx + 1}")
        axes[idx].grid(True)
        axes[idx].legend()
    plt.tight_layout()
    plt.show()


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate controller rollouts from dataset starting conditions.")
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=default_root / "simulator_config.yaml")
    parser.add_argument("--config-dir", type=Path, default=default_root / "configs")
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args(cli_args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)
    config_sim = load_yaml(args.config)
    config_model = load_yaml(resolve_saved_config(config_sim["model_path"], args.config_dir))
    results, payload = simulate_from_dataset(config_sim, config_model, args.project_root)

    energy_delta = np.array([item["sim_energy"] - item["ref_energy"] for item in results], dtype=np.float64)
    failed_runs = sum(1 for item in results if item["failed"])
    print(f"Mean energy delta: {energy_delta.mean():.6f} ± {energy_delta.std():.6f}")
    print(f"Failed runs: {failed_runs}/{len(results)}")

    if args.plot or config_sim.get("show_sim", False):
        _plot_payload(payload)


if __name__ == "__main__":
    main()
