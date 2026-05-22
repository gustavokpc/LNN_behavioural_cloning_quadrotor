#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Closed-loop rollout helpers that use the exported C controllers."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .c_controller import CController
from .quadrotor_sim import (
    LABEL_ALIASES,
    STATE_INDEX,
    integrate_state,
)


def build_c_input_vector(state: np.ndarray, input_labels: list[str], dt: float) -> np.ndarray:
    """Project the simulator state to the raw input order expected by the C export."""
    values: list[float] = []
    for label in input_labels:
        resolved = LABEL_ALIASES.get(label, label)
        if resolved == "omega":
            values.extend(float(state[STATE_INDEX[f"omega{i}"]]) for i in range(1, 5))
        elif resolved == "distance_error":
            values.append(float(np.linalg.norm(state[0:3])))
        elif resolved == "attitude_error":
            values.append(float(np.linalg.norm(state[6:9])))
        elif resolved in {"dt", "t"}:
            values.append(float(dt))
        else:
            if resolved not in STATE_INDEX:
                raise ValueError(f"Unsupported C-controller input label '{label}'.")
            values.append(float(state[STATE_INDEX[resolved]]))
    return np.asarray(values, dtype=np.float32)


def rollout_c_controller(controller: CController,
                         initial_state: np.ndarray,
                         input_labels: list[str],
                         dt: float,
                         horizon_steps: int,
                         integration_method: str = "explicit",
                         implicit_iters: int = 5,
                         stop_fn: Callable[[np.ndarray, int], bool] | None = None,
                         reset_controller: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the quadrotor dynamics using an exported C controller."""
    if reset_controller:
        controller.reset()

    state = np.asarray(initial_state, dtype=np.float64).copy()
    prev_deriv = None
    states = [state.copy()]
    actions = []

    for step_idx in range(int(horizon_steps)):
        controller_input = build_c_input_vector(state, input_labels, dt)
        action = np.clip(controller.predict(controller_input), 0.0, 1.0)
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

    return np.asarray(states), np.asarray(actions)
