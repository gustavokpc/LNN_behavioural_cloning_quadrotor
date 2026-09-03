"""Evaluation-only state capture and interventions around existing environments."""

from __future__ import annotations

from types import MethodType
from typing import Mapping

import numpy as np


def install_terminal_state_capture(env) -> None:
    """Capture true terminal arrays immediately before VecEnv auto-reset."""

    if hasattr(env, "_analysis_original_reset"):
        return
    original = env.reset_
    env._analysis_original_reset = original
    env.analysis_terminal_capture = {}

    def reset_with_capture(self, dones):
        dones = np.asarray(dones, dtype=bool)
        if np.any(dones):
            capture = {"mask": dones.copy()}
            for name in ("sim_states", "world_states", "states", "target_gates", "target_waypoints", "step_counts"):
                if hasattr(self, name):
                    capture[name] = np.asarray(getattr(self, name)).copy()
            self.analysis_terminal_capture = capture
        return self._analysis_original_reset(dones)

    env.reset_ = MethodType(reset_with_capture, env)


def true_state_snapshot(env, terminal: bool = False) -> tuple[np.ndarray, int]:
    """Return 19-state world-coordinate state and the active target index."""

    from ...utils.quadrotor_sim import body_to_world_state

    source = env.analysis_terminal_capture if terminal else None
    if hasattr(env, "sim_states"):
        sim_states = source["sim_states"] if source else env.sim_states
        target_gates = source["target_gates"] if source else env.target_gates
        target_idx = int(target_gates[0])
        target = env.gate_pos[target_idx % env.num_gates].astype(np.float64)
        state = body_to_world_state(sim_states[0].astype(np.float64))
        state[0:3] += target
        return state, target_idx
    states = source["states"] if source else env.states
    target_waypoints = source["target_waypoints"] if source else env.target_waypoints
    target_idx = int(target_waypoints[0])
    target = env.waypoints[target_idx % env.num_waypoints].astype(np.float64)
    state = body_to_world_state(states[0].astype(np.float64))
    state[0:3] += target
    return state, target_idx


def command_coordinates(env, policy_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized motor commands and commanded rotor speed coordinates."""

    action = np.asarray(policy_action, dtype=np.float64).reshape(1, -1)
    if hasattr(env, "action_to_bebop_command"):
        command = env.action_to_bebop_command(action)[0]
        parameters = env.dynamics_parameters[0]
        rpm = parameters.omega_min + command * (parameters.omega_max - parameters.omega_min)
        return command, rpm
    command = np.clip(action[0], 0.0, 1.0)
    from ...utils.dynamics_models import quadrotor_sim_matlab

    info = quadrotor_sim_matlab.INFO
    rpm = info.omega_min + command * (info.omega_max - info.omega_min)
    return command, rpm


def apply_initial_condition_scale(
    env,
    *,
    position: float = 1.0,
    velocity: float = 1.0,
    attitude: float = 1.0,
    angular_rate: float = 1.0,
) -> None:
    """Scale the already-sampled nominal reset dispersion in physical coordinates."""

    from ...utils.quadrotor_sim import body_to_world_state, world_to_body_state

    scales = (position, velocity, attitude, angular_rate)
    if any(value < 0.0 for value in scales):
        raise ValueError("Initial-condition scales must be non-negative.")
    if hasattr(env, "sim_states"):
        gate_idx = int(env.target_gates[0])
        target = env.gate_pos[gate_idx % env.num_gates].astype(np.float64)
        state = body_to_world_state(env.sim_states[0].astype(np.float64))
        absolute = target + state[0:3]
        nominal_position = env.start_pos.astype(np.float64)
        absolute = nominal_position + position * (absolute - nominal_position)
        state[0:3] = absolute - target
        state[3:6] *= velocity
        state[6:9] *= attitude
        state[8] = (state[8] + np.pi) % (2.0 * np.pi) - np.pi
        state[9:12] *= angular_rate
        env.sim_states[0] = world_to_body_state(state).astype(np.float32)
        env.update_states_gate()
        return
    target_idx = int(env.target_waypoints[0])
    target = env.waypoints[target_idx % env.num_waypoints].astype(np.float64)
    state = body_to_world_state(env.states[0].astype(np.float64))
    absolute = target + state[0:3]
    nominal_position = target if bool(getattr(env, "point_to_point", False)) else env.start_pos.astype(np.float64)
    state[0:3] = nominal_position + position * (absolute - nominal_position) - target
    state[3:6] *= velocity
    state[6:9] *= attitude
    state[8] = (state[8] + np.pi) % (2.0 * np.pi) - np.pi
    state[9:12] *= angular_rate
    env.states[0] = world_to_body_state(state).astype(np.float32)


def apply_state_perturbation(env, perturbation: Mapping[str, object]) -> None:
    """Apply an instantaneous physical-state perturbation without changing dynamics."""

    from ...utils.quadrotor_sim import body_to_world_state, world_to_body_state

    body = env.sim_states[0] if hasattr(env, "sim_states") else env.states[0]
    state = body_to_world_state(body.astype(np.float64))
    fields = {
        "position": slice(0, 3),
        "velocity": slice(3, 6),
        "attitude": slice(6, 9),
        "angular_rate": slice(9, 12),
        "external_moment": slice(12, 15),
    }
    for name, section in fields.items():
        if name in perturbation:
            delta = np.asarray(perturbation[name], dtype=np.float64)
            if delta.shape != (3,):
                raise ValueError(f"{name} perturbation must contain three values.")
            state[section] += delta
    body_new = world_to_body_state(state).astype(np.float32)
    if hasattr(env, "sim_states"):
        env.sim_states[0] = body_new
        env.update_states_gate()
    else:
        env.states[0] = body_new
