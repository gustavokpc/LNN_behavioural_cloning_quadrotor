#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np

from utils.animation import animate
from utils.config import load_yaml, resolve_saved_config
from simulators.Simulator_start_dataset import _initial_window, _prepare_dataset
from utils.quadrotor_sim import (
    body_to_world_trajectory,
    input_labels_without_time,
    rollout_controller,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a dataset-based quadrotor rollout with the animation helper."
    )
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--config",
        type=Path,
        default=default_root / "simulator_config.yaml",
        help="Path to the simulator config YAML file.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=default_root / "configs",
        help="Directory containing saved model configs.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="Project root directory used to resolve checkpoint paths.",
    )
    parser.add_argument(
        "--trajectory",
        type=int,
        default=0,
        help="Index of the dataset trajectory to simulate and animate.",
    )
    parser.add_argument(
        "--trajectories",
        type=int,
        default=1,
        help="Number of dataset trajectories to simulate and animate together.",
    )
    parser.add_argument(
        "--simultaneous",
        action="store_true",
        help="Draw multiple trajectories at once in the same animation window.",
    )
    parser.add_argument(
        "--draw-path",
        action="store_true",
        help="Draw the trajectory path of each drone in the animation.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the animation to an MP4 file instead of showing it interactively.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rollout.mp4",
        help="Output filename when --record is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_sim = load_yaml(args.config)
    config_model = load_yaml(resolve_saved_config(config_sim["model_path"], args.config_dir))

    raw_inputs, raw_outputs, dt_values = _prepare_dataset(config_model, str(config_sim["dataset"]["path_sim"]))

    if args.trajectory < 0 or args.trajectory >= raw_inputs.shape[0]:
        raise ValueError(
            f"Trajectory index {args.trajectory} is out of range [0, {raw_inputs.shape[0] - 1}]."
        )
    if args.trajectories < 1:
        raise ValueError("--trajectories must be at least 1.")
    if args.trajectory + args.trajectories > raw_inputs.shape[0]:
        raise ValueError(
            f"Requested {args.trajectories} trajectories starting at {args.trajectory} exceeds available dataset size {raw_inputs.shape[0]}."
        )

    use_sequencing = bool(config_model.get("sequencing", {}).get("value", False))
    seq_len = int(config_model.get("sequencing", {}).get("seq_len", 1))
    normalization_limits = config_model["dataset"].get(
        "normalization_limits", config_model["dataset"].get("bebop_model", "bebop1")
    )
    model = _build_model(config_model, args.project_root, config_sim["model_path"])

    times = []
    xs = []
    ys = []
    zs = []
    phis = []
    thetas = []
    psis = []
    us = []
    names = []

    for traj_idx in range(args.trajectory, args.trajectory + args.trajectories):
        dt = float(dt_values[traj_idx])
        raw_traj_inputs = raw_inputs[traj_idx]
        initial_state = _prepare_initial_state(raw_traj_inputs, config_model["dataset"]["input_labels"])
        init_window = _initial_window(
            raw_traj_inputs,
            config_model["dataset"]["input_labels"],
            seq_len,
            normalization_limits,
        )

        simulated_states_body, generated_actions = rollout_controller(
            model=model,
            initial_state=initial_state,
            input_labels=config_model["dataset"]["input_labels"],
            dt=dt,
            horizon_steps=max(1, int(round(config_sim["simulation"]["time_simulation"] / dt))),
            device=_get_device(),
            use_sequencing=use_sequencing,
            seq_len=seq_len,
            initial_window=init_window,
            force_timespans=True,
            integration_method=config_sim["simulation"].get("integration_method", "explicit"),
            implicit_iters=int(config_sim["simulation"].get("implicit_iterations", 5)),
        )

        world_states = body_to_world_trajectory(simulated_states_body)
        steps = min(len(world_states), len(generated_actions))
        world_states = world_states[:steps]
        actions = generated_actions[:steps]
        t = np.arange(steps, dtype=np.float64) * float(config_sim["simulation"]["dt"])

        times.append(t)
        xs.append(world_states[:, 0])
        ys.append(world_states[:, 1])
        zs.append(world_states[:, 2])
        phis.append(world_states[:, 6])
        thetas.append(world_states[:, 7])
        psis.append(world_states[:, 8])
        us.append(actions)
        names.append(f"traj_{traj_idx}")

    max_steps = max(arr.shape[0] for arr in xs)
    def pad_sequence(arr, length, axis=0):
        if arr.shape[0] >= length:
            return arr
        pad_shape = list(arr.shape)
        pad_shape[0] = length - arr.shape[0]
        padding = np.repeat(arr[-1:], repeats=length - arr.shape[0], axis=axis)
        return np.concatenate([arr, padding], axis=axis)

    xs = [pad_sequence(arr, max_steps) for arr in xs]
    ys = [pad_sequence(arr, max_steps) for arr in ys]
    zs = [pad_sequence(arr, max_steps) for arr in zs]
    phis = [pad_sequence(arr, max_steps) for arr in phis]
    thetas = [pad_sequence(arr, max_steps) for arr in thetas]
    psis = [pad_sequence(arr, max_steps) for arr in psis]
    us = [pad_sequence(arr, max_steps, axis=0) for arr in us]

    base_colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 128, 255),
        (128, 255, 128),
        (255, 128, 128),
    ]
    colors = [base_colors[i % len(base_colors)] for i in range(len(names))]

    animate(
        t=times,
        x=np.stack(xs, axis=0),
        y=np.stack(ys, axis=0),
        z=np.stack(zs, axis=0),
        phi=np.stack(phis, axis=0),
        theta=np.stack(thetas, axis=0),
        psi=np.stack(psis, axis=0),
        u=np.stack(us, axis=0),
        autopilot_mode=[],
        target=[],
        waypoints=[],
        file=args.output,
        multiple_trajectories=True,
        simultaneous=args.simultaneous,
        colors=colors,
        names=names,
        record=args.record,
        draw_path=args.draw_path or args.simultaneous,
    )


def _get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(config_model: dict, project_root: Path, model_path: str):
    from utils.quadrotor_sim import build_lightning_model
    return build_lightning_model(config_model, model_path, project_root, _get_device())


def _prepare_initial_state(raw_traj_inputs: np.ndarray, input_labels: list[str]) -> np.ndarray:
    from utils.quadrotor_sim import state_from_input_features
    from utils.data import expand_feature_labels
    base_labels = [label for label in input_labels if label not in {"t", "dt"}]
    expanded_labels = expand_feature_labels(base_labels)
    return state_from_input_features(raw_traj_inputs[:, 0], expanded_labels)


if __name__ == "__main__":
    main()
