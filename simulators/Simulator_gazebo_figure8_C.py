#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gazebo/Paparazzi figure-eight waypoint simulation with C-exported controllers."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np

from simulators.Simulator_gazebo_square_C import (
    MODEL_PRESETS,
    _animate_square,
    _enu_to_network_world,
    _network_world_to_enu,
    _wrap_angle,
    resolve_model_paths,
)
from utils.c_controller import CController
from utils.config import load_yaml
from utils.dynamics_models import available_dynamics_models, get_dynamics_info, set_dynamics_model
from utils.quadrotor_sim import body_to_world_trajectory, world_to_body_state
from utils.quadrotor_sim_c import rollout_c_controller


DEFAULT_FLIGHT_PLAN = (
    "/home/gustavokpc/Documents/ESTAG/paparazzi_mavlab/paparazzi/"
    "conf/flight_plans/tudelft/rl_cfc_waypoints_square.xml"
)


def _waypoint_height(elem: ET.Element, default_alt: float) -> float:
    if "height" in elem.attrib:
        return float(elem.attrib["height"])
    if "alt" in elem.attrib:
        return float(elem.attrib["alt"])
    return default_alt


def _load_flight_plan_figure8(
    path: Path,
    start_alt_m: float = 1.0,
    waypoint_alt_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    default_alt = float(root.attrib.get("alt", waypoint_alt_m))
    waypoint_root = root.find("waypoints")
    if waypoint_root is None:
        raise KeyError(f"Flight plan has no <waypoints> section: {path}")

    waypoints: dict[str, np.ndarray] = {}
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
        figure8 = np.asarray([waypoints[f"RL_F8_{idx}"] for idx in range(1, 9)], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(f"Flight plan is missing waypoint {exc!s}.") from exc

    standby[2] = float(start_alt_m)
    figure8[:, 2] = float(waypoint_alt_m)
    return standby, figure8


def simulate_gazebo_figure8(
    config_model: dict,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    set_dynamics_model(dynamics_model)
    dynamics_info = get_dynamics_info()
    standby_enu, figure8_enu = _load_flight_plan_figure8(
        flight_plan,
        start_alt_m=start_alt_m,
        waypoint_alt_m=waypoint_alt_m,
    )
    waypoints_world = np.asarray([_enu_to_network_world(wp) for wp in figure8_enu], dtype=np.float64)

    current_world = np.zeros(19, dtype=np.float64)
    current_world[0:3] = _enu_to_network_world(standby_enu)
    current_world[15:19] = dynamics_info.omega_mid

    controller = CController(c_model_dir)
    controller.reset()
    max_total_steps = int(round(time_simulation / dt))
    waypoint_index = start_waypoint_index % len(waypoints_world)
    elapsed_steps = 0
    waypoints_passed = 0
    total_states: list[np.ndarray] = []
    total_actions: list[np.ndarray] = []
    target_trace: list[np.ndarray] = []

    initial_distance = float(np.linalg.norm(standby_enu - figure8_enu[waypoint_index]))
    print(f"Initial ENU position STDBY: {standby_enu}")
    print(f"First target RL_F8_{waypoint_index + 1}: {figure8_enu[waypoint_index]}")
    print(f"Initial distance: {initial_distance:.3f} m")
    print(f"Waypoint switch distance: {dist_error:.3f} m")
    hover = "n/a" if dynamics_info.hover_omega is None else f"{dynamics_info.hover_omega:.3f} RPM"
    u_hover = "n/a" if dynamics_info.u_hover is None else f"{dynamics_info.u_hover:.6f}"
    print(f"Dynamics model: {dynamics_info.name} | hover={hover} | u_hover={u_hover}")
    print(f"Reset each waypoint: {reset_each_waypoint}")

    while elapsed_steps < max_total_steps:
        target = waypoints_world[waypoint_index]
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
        waypoints_passed += 1
        waypoint_index = (waypoint_index + 1) % len(waypoints_world)

    states = np.vstack(total_states)
    actions = np.vstack(total_actions) if total_actions else np.zeros((0, 4), dtype=np.float64)
    targets = np.asarray(target_trace[: len(states)], dtype=np.float64)
    return states, actions, targets, waypoints_passed, standby_enu, figure8_enu


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Simulate the Paparazzi RL figure-eight with a C-exported model.")
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
    parser.add_argument("--start-waypoint-index", type=int, default=0)
    parser.add_argument("--start-alt", type=float, default=1.0)
    parser.add_argument("--waypoint-alt", type=float, default=1.5)
    parser.add_argument("--reset-each-waypoint", action="store_true")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", default="gazebo_figure8_cfc.mp4")
    parser.add_argument("--auto-play", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(cli_args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)
    model_config, c_model_dir = resolve_model_paths(args.project_root, args.model, args.model_config, args.c_model_dir)
    config_model = load_yaml(model_config)
    states_world, actions, targets_world, waypoints_passed, standby_enu, figure8_enu = simulate_gazebo_figure8(
        config_model=config_model,
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
    print(f"Waypoints passed: {waypoints_passed}")
    print(f"Total time: {total_time:.3f} s")
    print(f"Total energy: {total_energy:.3f}")
    print(f"Final ENU position: {final_enu}")
    print(f"Figure-eight waypoints ENU: {figure8_enu}")
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
            square_enu=figure8_enu,
            dt=args.dt,
            record=args.record,
            output=args.output,
            auto_play=args.auto_play,
        )


if __name__ == "__main__":
    main()
