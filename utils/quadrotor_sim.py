#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared quadrotor simulation helpers for controller rollouts.

This module centralizes the repeated simulation logic that used to live in
multiple standalone scripts: state transforms, continuous-time dynamics,
discretization schemes, checkpoint-backed model loading, and closed-loop rollout.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .config import dataset_dims, resolve_checkpoint
from .data import get_norm_vectors
from .model_builder import build_controller_network
from .lightning import Lightning_Model
from .dynamics_models import dynamics as selected_dynamics
from .dynamics_models import get_dynamics_info


STATE_LABELS = [
    "dx", "dy", "dz",
    "vx", "vy", "vz",
    "phi", "theta", "psi",
    "p", "q", "r",
    "Mx_ext", "My_ext", "Mz_ext",
    "omega1", "omega2", "omega3", "omega4",
]
STATE_INDEX = {label: idx for idx, label in enumerate(STATE_LABELS)}
LABEL_ALIASES = {"Mx": "Mx_ext", "My": "My_ext", "Mz": "Mz_ext"}


def input_labels_without_time(input_labels: list[str]) -> list[str]:
    """Drop explicit time features when building simulator state vectors."""
    return [label for label in input_labels if label not in {"t", "dt"}]


def rotation_body_to_world(phi: float, theta: float, psi: float) -> np.ndarray:
    """Build the body-to-world rotation matrix from Euler angles."""
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]])
    ry = np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]])
    rz = np.array([[np.cos(psi), -np.sin(psi), 0.0], [np.sin(psi), np.cos(psi), 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def world_to_body_state(state_world: np.ndarray) -> np.ndarray:
    """Convert position/velocity entries of a world-frame state into body frame."""
    state_world = np.asarray(state_world, dtype=np.float64).copy()
    x, y, z, vx, vy, vz, phi, theta, psi = state_world[:9]
    rotation = rotation_body_to_world(phi, theta, psi)
    state_world[0:3] = -rotation.T @ np.array([x, y, z])
    state_world[3:6] = rotation.T @ np.array([vx, vy, vz])
    return state_world


def body_to_world_state(state_body: np.ndarray) -> np.ndarray:
    """Convert position/velocity entries of a body-frame state back to world frame."""
    state_body = np.asarray(state_body, dtype=np.float64).copy()
    dx, dy, dz, vx, vy, vz, phi, theta, psi = state_body[:9]
    rotation = rotation_body_to_world(phi, theta, psi)
    state_body[0:3] = -rotation @ np.array([dx, dy, dz])
    state_body[3:6] = rotation @ np.array([vx, vy, vz])
    return state_body


def body_to_world_trajectory(states_body: np.ndarray) -> np.ndarray:
    return np.asarray([body_to_world_state(state) for state in states_body], dtype=np.float64)


def dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Evaluate the continuous-time quadrotor dynamics for one state/action pair."""
    return selected_dynamics(state, action)


def integrate_state(method: str,
                    state: np.ndarray,
                    action: np.ndarray,
                    dt: float,
                    prev_deriv: np.ndarray | None = None,
                    implicit_iters: int = 5) -> tuple[np.ndarray, np.ndarray]:
    # Each simulator script can choose the integrator through YAML without
    # having to duplicate the discretization implementation.
    method = method.lower()
    deriv = dynamics(state, action)

    if method in {"explicit", "euler"}:
        return state + dt * deriv, deriv

    if method == "rk4":
        k1 = deriv
        k2 = dynamics(state + 0.5 * dt * k1, action)
        k3 = dynamics(state + 0.5 * dt * k2, action)
        k4 = dynamics(state + dt * k3, action)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), deriv

    if method == "adams-bashforth":
        if prev_deriv is None:
            return state + dt * deriv, deriv
        return state + dt * (1.5 * deriv - 0.5 * prev_deriv), deriv

    if method == "adams-moulton":
        if prev_deriv is None:
            return state + dt * deriv, deriv
        predictor = state + dt * (1.5 * deriv - 0.5 * prev_deriv)
        return state + 0.5 * dt * (dynamics(predictor, action) + deriv), deriv

    if method == "implicit":
        next_state = state.copy()
        for _ in range(max(1, int(implicit_iters))):
            next_state = state + dt * dynamics(next_state, action)
        return next_state, deriv

    raise ValueError(
        f"Unsupported integration method '{method}'. "
        "Choose from explicit, euler, rk4, adams-bashforth, adams-moulton, implicit."
    )


def build_input_vector(state: np.ndarray, input_labels: list[str]) -> np.ndarray:
    """Project a full simulator state vector down to the configured model inputs."""
    values: list[float] = []
    for label in input_labels:
        resolved = LABEL_ALIASES.get(label, label)
        if resolved == "omega":
            values.extend(state[STATE_INDEX[f"omega{i}"]] for i in range(1, 5))
        elif resolved == "distance_error":
            values.append(float(np.linalg.norm(state[0:3])))
        elif resolved == "attitude_error":
            values.append(float(np.linalg.norm(state[6:9])))
        else:
            if resolved not in STATE_INDEX:
                raise ValueError(f"Unsupported simulation input label '{label}'.")
            values.append(float(state[STATE_INDEX[resolved]]))
    return np.asarray(values, dtype=np.float64)


def state_from_input_features(values: np.ndarray, input_labels: list[str]) -> np.ndarray:
    """
    Reconstruct the 19-state body-frame vector from dataset features.

    Missing state entries are left at zero, which keeps the helper usable for
    ablated or reduced-feature models while still allowing the simulator to run.
    """
    state = np.zeros(len(STATE_LABELS), dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    cursor = 0
    for label in input_labels:
        resolved = LABEL_ALIASES.get(label, label)
        if resolved in {"distance_error", "attitude_error"}:
            cursor += 1
            continue
        if resolved == "omega":
            state[STATE_INDEX["omega1"]:STATE_INDEX["omega4"] + 1] = values[cursor:cursor + 4]
            cursor += 4
            continue
        if resolved in STATE_INDEX:
            state[STATE_INDEX[resolved]] = values[cursor]
        cursor += 1
    return state


def normalize_input(values: np.ndarray, input_labels: list[str]) -> np.ndarray:
    """Normalize simulator inputs with the same bounds used for training data."""
    global_min, global_max = get_norm_vectors(input_labels)
    global_min = global_min.reshape(-1)
    global_max = global_max.reshape(-1)
    return (values - global_min) / (global_max - global_min + 1e-10)


def build_normalized_window(state: np.ndarray, input_labels: list[str], seq_len: int) -> np.ndarray:
    """Create the initial normalized history window for sequence models."""
    obs = normalize_input(build_input_vector(state, input_labels), input_labels)
    return np.repeat(obs[np.newaxis, :], repeats=max(1, seq_len), axis=0)


def update_observation_window(window: np.ndarray, state: np.ndarray, input_labels: list[str], seq_len: int) -> np.ndarray:
    """Append a newly simulated state to the rolling observation history."""
    new_obs = normalize_input(build_input_vector(state, input_labels), input_labels)
    if seq_len <= 1:
        return new_obs[np.newaxis, :]
    return np.concatenate([window[-(seq_len - 1):], new_obs[np.newaxis, :]], axis=0)


def make_model_observation(window: np.ndarray, use_sequencing: bool, device: torch.device) -> torch.Tensor:
    """Convert a NumPy history window into the tensor layout expected by the model."""
    if use_sequencing:
        obs = window[np.newaxis, np.newaxis, :, :]
    else:
        obs = window[-1:, :][np.newaxis, :, :]
    return torch.tensor(obs, dtype=torch.float32, device=device)


def build_lightning_model(config_model: dict, checkpoint_path: str | None, project_root, device: torch.device):
    """Rebuild a controller and optionally load its checkpoint weights."""
    input_dim, output_dim = dataset_dims(config_model)
    network = build_controller_network(config_model, input_dim, output_dim)
    model = Lightning_Model(network, config_model)
    if checkpoint_path is not None:
        ckpt = resolve_checkpoint(checkpoint_path, project_root)
        state_dict = torch.load(ckpt, map_location=device, weights_only=True)["state_dict"]
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def rollout_controller(model,
                       initial_state: np.ndarray,
                       input_labels: list[str],
                       dt: float,
                       horizon_steps: int,
                       device: torch.device,
                       use_sequencing: bool,
                       seq_len: int,
                       initial_window: np.ndarray | None = None,
                       force_timespans: bool = False,
                       integration_method: str = "explicit",
                       implicit_iters: int = 5,
                       stop_fn: Callable[[np.ndarray, int], bool] | None = None) -> tuple[np.ndarray, np.ndarray]:
    # The rollout helper is shared by all simulator entrypoints so recurrent
    # state handling and observation-window updates stay identical everywhere.
    base_labels = input_labels_without_time(input_labels)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    window = np.array(initial_window, copy=True) if initial_window is not None else build_normalized_window(state, base_labels, seq_len)
    dt_tensor = None
    if force_timespans or getattr(model, "with_time", False):
        dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device).reshape(1, 1, 1)

    hx = None
    prev_deriv = None
    states = [state.copy()]
    actions = []

    for step_idx in range(int(horizon_steps)):
        obs_tensor = make_model_observation(window, use_sequencing=use_sequencing, device=device)
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
        window = update_observation_window(window, state, base_labels, seq_len)

    return np.asarray(states), np.asarray(actions)


def generate_starting_conditions(n_conditions: int, seed: int = 42) -> np.ndarray:
    """Sample randomized but physically plausible landing start conditions."""
    dynamics_info = get_dynamics_info()
    rng = np.random.default_rng(seed)
    starts = np.zeros((n_conditions, 19), dtype=np.float64)
    starts[:, 0:2] = rng.choice([-1, 1], size=(n_conditions, 2)) * rng.uniform(1.0, 5.0, size=(n_conditions, 2))
    starts[:, 2] = rng.uniform(-1.0, 1.0, size=n_conditions)
    starts[:, 3:6] = rng.uniform(-0.5, 0.5, size=(n_conditions, 3))
    starts[:, 6:8] = rng.uniform(-2.0 * np.pi / 9.0, 2.0 * np.pi / 9.0, size=(n_conditions, 2))
    starts[:, 8] = rng.uniform(-np.pi, np.pi, size=n_conditions)
    starts[:, 9:12] = rng.uniform(-1.0, 1.0, size=(n_conditions, 3))
    starts[:, 12:15] = rng.uniform(-0.01, 0.01, size=(n_conditions, 3))
    starts[:, 15:19] = dynamics_info.omega_mid
    return starts
