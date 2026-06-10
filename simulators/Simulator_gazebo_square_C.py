#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gazebo/Paparazzi square-waypoint simulation using the exported CFC controller."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np

from utils.animation import animate
from utils.c_controller import CController
from utils.config import load_yaml
from utils.dynamics_models import available_dynamics_models, get_dynamics_info, set_dynamics_model
from utils.quadrotor_sim import body_to_world_trajectory, world_to_body_state
from utils.quadrotor_sim_c import rollout_c_controller


MODEL_PRESETS = {
    "MLP": ("configs/mlp_epoch=19_val_loss=0.003130.yaml", "C_codes/MLP"),
    "LTC": ("configs/LTC_64_neurons_seq_1_epoch=18_val_loss=0.000193.yaml", "C_codes/LTC"),
    "RNN": ("configs/RNN_64_neurons_seq_1_epoch=17_val_loss=0.000147.yaml", "C_codes/RNN"),
    "CONV_CFC_DEFAULT": ("configs/conv_cfc_default_n64_epoch=17_val_loss=0.000326.yaml", "C_codes/CONV_CFC_DEFAULT"),
    "CFC": ("configs/new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.yaml", "C_codes/CFC"),
    "CFC_PURE": ("configs/new_CFC_pure_64_neurons_seq_1_epoch=17_val_loss=0.000203.yaml", "C_codes/CFC_PURE"),
    "CTRNN": ("configs/new_CTRNN_64_neurons_seq_1_epoch=19_val_loss=0.000150.yaml", "C_codes/CTRNN"),
    "GRU": ("configs/new_GRU_64_neurons_seq_1_epoch=19_val_loss=0.000088.yaml", "C_codes/GRU"),
    "LSTM": ("configs/new_LSTM_64_neurons_seq_1_epoch=17_val_loss=0.000092.yaml", "C_codes/LSTM"),
    "NCP_CFC": ("configs/new_NCP_CFC_60_neurons_seq_1_epoch=18_val_loss=0.000143.yaml", "C_codes/NCP_CFC"),
}
DEFAULT_FLIGHT_PLAN = (
    "/home/gustavokpc/Documents/ESTAG/paparazzi/"
    "conf/flight_plans/tudelft/nn_waypoints_square.xml"
)


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


def _waypoint_height(elem: ET.Element, default_alt: float) -> float:
    if "height" in elem.attrib:
        return float(elem.attrib["height"])
    if "alt" in elem.attrib:
        return float(elem.attrib["alt"])
    return default_alt


def _load_flight_plan_square(
    path: Path,
    start_alt_m: float = 1.0,
    waypoint_alt_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    default_alt = float(root.attrib.get("alt", 1.0))
    waypoints: dict[str, np.ndarray] = {}
    waypoint_root = root.find("waypoints")
    if waypoint_root is None:
        raise KeyError(f"Flight plan has no <waypoints> section: {path}")
    for elem in waypoint_root:
        name = elem.attrib.get("name")
        if not name or "x" not in elem.attrib or "y" not in elem.attrib:
            continue
        waypoints[name] = np.asarray(
            [
                float(elem.attrib["x"]),
                float(elem.attrib["y"]),
                _waypoint_height(elem, default_alt),
            ],
            dtype=np.float64,
        )

    try:
        standby = waypoints["STDBY"].copy()
        square = np.asarray([waypoints[f"NN_SQ_{idx}"] for idx in range(1, 5)], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(f"Flight plan is missing waypoint {exc!s}.") from exc
    standby[2] = float(start_alt_m)
    square[:, 2] = float(waypoint_alt_m)
    return standby, square


def _wrap_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


def simulate_gazebo_square(
    config_model: dict,
    project_root: Path,
    flight_plan: Path,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    set_dynamics_model(dynamics_model)
    dynamics_info = get_dynamics_info()
    standby_enu, square_enu = _load_flight_plan_square(
        flight_plan,
        start_alt_m=start_alt_m,
        waypoint_alt_m=waypoint_alt_m,
    )
    square_world = np.asarray([_enu_to_network_world(wp) for wp in square_enu], dtype=np.float64)
    current_world = np.zeros(19, dtype=np.float64)
    current_world[0:3] = _enu_to_network_world(standby_enu)
    current_world[15:19] = dynamics_info.omega_mid

    controller = CController(c_model_dir)
    controller.reset()
    max_total_steps = int(round(time_simulation / dt))
    gate_index = start_waypoint_index % len(square_world)
    elapsed_steps = 0
    gates_passed = 0
    total_states: list[np.ndarray] = []
    total_actions: list[np.ndarray] = []
    target_trace: list[np.ndarray] = []

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
            input_labels=config_model["dataset"]["input_labels"],
            dt=dt,
            horizon_steps=max_total_steps - elapsed_steps,
            integration_method=integration_method,
            implicit_iters=implicit_iters,
            stop_fn=lambda state, step: np.linalg.norm(state[0:3]) < dist_error,
            reset_controller=reset_each_waypoint,
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


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Simulate the Paparazzi Gazebo square with the CFC C export.")
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--flight-plan", type=Path, default=Path(DEFAULT_FLIGHT_PLAN))
    parser.add_argument("--model", default="CFC", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--c-model-dir", type=Path, default=None)
    parser.add_argument("--dynamics-model", default="quadrotor_sim_matlab", choices=available_dynamics_models())
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--time-simulation", type=float, default=60.0)
    parser.add_argument("--dist-error", type=float, default=0.1)
    parser.add_argument("--integration-method", default="rk4")
    parser.add_argument("--implicit-iters", type=int, default=1)
    parser.add_argument("--start-waypoint-index", type=int, default=3)
    parser.add_argument("--start-alt", type=float, default=1.0)
    parser.add_argument("--waypoint-alt", type=float, default=1.5)
    parser.add_argument("--reset-each-waypoint", action="store_true")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", default="gazebo_square_cfc.mp4")
    parser.add_argument("--auto-play", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(cli_args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)
    model_config, c_model_dir = resolve_model_paths(args.project_root, args.model, args.model_config, args.c_model_dir)
    config_model = load_yaml(model_config)
    standby_enu, square_enu = _load_flight_plan_square(
        args.flight_plan,
        start_alt_m=args.start_alt,
        waypoint_alt_m=args.waypoint_alt,
    )
    states_world, actions, targets_world, gates_passed = simulate_gazebo_square(
        config_model=config_model,
        project_root=args.project_root,
        flight_plan=args.flight_plan,
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
