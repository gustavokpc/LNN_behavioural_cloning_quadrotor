"""Exploratory sample-local stability diagnostics (never global certificates)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .environment import install_terminal_state_capture, true_state_snapshot
from .io import read_json, write_json
from .loading import load_model_and_env
from .rollout import load_steps


def central_difference_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, float]:
    """Central Jacobian plus a forward/central discrepancy sanity residual."""

    point = np.asarray(point, dtype=np.float64)
    baseline = np.asarray(function(point), dtype=np.float64)
    jacobian = np.empty((baseline.size, point.size), dtype=np.float64)
    forward = np.empty_like(jacobian)
    for column in range(point.size):
        delta = np.zeros_like(point)
        delta[column] = epsilon
        plus = np.asarray(function(point + delta), dtype=np.float64)
        minus = np.asarray(function(point - delta), dtype=np.float64)
        jacobian[:, column] = (plus - minus) / (2.0 * epsilon)
        forward[:, column] = (plus - baseline) / epsilon
    residual = float(np.linalg.norm(jacobian - forward) / max(np.linalg.norm(jacobian), 1e-12))
    return jacobian, residual


def spectral_summary(jacobian: np.ndarray) -> dict[str, object]:
    eigenvalues = np.linalg.eigvals(np.asarray(jacobian, dtype=np.float64))
    return {
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "eigenvalues_real": eigenvalues.real.tolist(),
        "eigenvalues_imag": eigenvalues.imag.tolist(),
        "interpretation": "Sample-local discrete linearization; radius < 1 indicates contraction only for this linearization.",
    }


def finite_time_contraction(jacobians: Iterable[np.ndarray], dt: float) -> dict[str, float]:
    jacobians = [np.asarray(jacobian, dtype=np.float64) for jacobian in jacobians]
    if not jacobians:
        raise ValueError("At least one Jacobian is required.")
    product = np.eye(jacobians[0].shape[0])
    for jacobian in jacobians:
        product = jacobian @ product
    radius = float(np.max(np.abs(np.linalg.eigvals(product))))
    return {
        "window_steps": len(jacobians),
        "window_seconds": len(jacobians) * dt,
        "product_spectral_radius": radius,
        "effective_log_rate_per_second": float(np.log(max(radius, 1e-300)) / (len(jacobians) * dt)),
    }


def common_quadratic_lyapunov(jacobians: Iterable[np.ndarray], margin: float = 1e-6) -> dict[str, object]:
    """Try a common local discrete-time quadratic Lyapunov LMI via cvxpy."""

    try:
        import cvxpy as cp
    except ImportError:
        return {"supported": False, "reason": "cvxpy is not installed"}
    matrices = [np.asarray(matrix, dtype=np.float64) for matrix in jacobians]
    if not matrices:
        return {"supported": False, "reason": "no local Jacobians supplied"}
    n = matrices[0].shape[0]
    p = cp.Variable((n, n), symmetric=True)
    constraints = [p >> margin * np.eye(n)]
    constraints.extend(matrix.T @ p @ matrix - p << -margin * np.eye(n) for matrix in matrices)
    problem = cp.Problem(cp.Minimize(cp.trace(p)), constraints)
    try:
        problem.solve()
    except cp.error.SolverError as exc:
        return {"supported": True, "status": "solver_error", "reason": str(exc)}
    return {
        "supported": True,
        "status": problem.status,
        "interpretation": "Feasibility applies only to the supplied family of sampled local linear models; infeasibility is not proof of instability.",
    }


def recurrent_hidden_jacobian(model, observation: np.ndarray, hidden: np.ndarray | None = None) -> np.ndarray:
    """Autograd Jacobian dh_next/dh at fixed observation for the actor cell."""

    import torch

    policy = model.policy
    recurrent = getattr(policy, "lstm_actor", None)
    if recurrent is None:
        raise ValueError("Policy has no recurrent actor state.")
    obs_tensor, _ = policy.obs_to_tensor(np.asarray(observation, dtype=np.float32).reshape(1, -1))
    features = policy.extract_features(obs_tensor)
    if isinstance(features, tuple):
        features = features[0]
    size = int(getattr(recurrent, "state_size", getattr(recurrent, "hidden_size", 0)))
    base = torch.zeros(size, dtype=features.dtype, device=features.device) if hidden is None else torch.as_tensor(hidden, dtype=features.dtype, device=features.device)

    def transition(value):
        state = value.reshape(1, -1)
        if recurrent.__class__.__name__ in {"CfC", "LTC"}:
            timespan = torch.as_tensor(float(getattr(recurrent, "cfc_timespan", 1.0)), device=value.device, dtype=value.dtype).reshape(1, 1, 1)
            _, next_hidden = recurrent(features.unsqueeze(1), state, timespans=timespan)
        else:
            _, next_hidden = recurrent(features.unsqueeze(1), state.unsqueeze(0))
            if isinstance(next_hidden, tuple):
                next_hidden = next_hidden[0]
        return next_hidden.reshape(-1)

    return torch.autograd.functional.jacobian(transition, base).detach().cpu().numpy()


def analyze_recurrent_stability(
    result_dir: str | Path,
    *,
    sample_index: int = 0,
    epsilon_values: Iterable[float] = (1e-4, 1e-5, 1e-6),
    device: str = "cpu",
) -> dict[str, object]:
    """Analyse dh_next/dh at one saved point and report epsilon-independent autograd result."""

    result_dir = Path(result_dir)
    metadata = read_json(result_dir / "metadata.json")
    model, env, _, _ = load_model_and_env(
        metadata["checkpoint_path"], architecture=metadata["architecture"], device=device,
        seed=int(metadata["evaluation_seed"]), policy_dt=metadata.get("policy_dt"),
        env_overrides=dict(metadata.get("environment", {})),
    )
    env.close()
    steps = load_steps(result_dir)
    if sample_index < 0 or sample_index >= len(steps["policy_observation"]):
        raise IndexError("sample_index outside saved rollout.")
    same_episode = sample_index > 0 and steps["episode_id"][sample_index - 1] == steps["episode_id"][sample_index]
    hidden = steps["hidden_state"][sample_index - 1] if same_episode and steps["hidden_state"].shape[1] else None
    jacobian = recurrent_hidden_jacobian(model, steps["policy_observation"][sample_index], hidden)
    result = {
        "scope": "local recurrent actor-state map at fixed saved observation",
        "sample_index": sample_index,
        "shape": list(jacobian.shape),
        **spectral_summary(jacobian),
    }
    np.savez_compressed(result_dir / "recurrent_stability.npz", jacobian=jacobian)
    write_json(result_dir / "recurrent_stability.json", result)
    return result


def analyze_closed_loop_stability(
    result_dir: str | Path,
    *,
    sample_index: int = 0,
    epsilon_values: Iterable[float] = (1e-4, 1e-5),
    device: str = "cpu",
) -> dict[str, object]:
    """Finite-difference the actual one-step plant+policy+actor-hidden map.

    The discrete gate/waypoint target is held at the saved value. Samples whose
    nominal or perturbed step crosses a gate, terminates, or truncates are rejected.
    Observation/action histories and AB2 derivative memory are rejected because
    those extra Markov variables are not yet serialized by the environment.
    """

    from ...utils.quadrotor_sim import world_to_body_state

    result_dir = Path(result_dir)
    metadata = read_json(result_dir / "metadata.json")
    environment = metadata.get("environment", {})
    if int(environment.get("num_state_history", 0)) or int(environment.get("num_action_history", 0)):
        raise ValueError("Closed-loop Jacobian unsupported with observation/action history: saved Markov state is incomplete.")
    if str(environment.get("integration_method", "rk4")).lower() in {"ab2", "adams_bashforth_2"}:
        raise ValueError("Closed-loop Jacobian unsupported for AB2 because previous derivative is not in steps.npz.")
    model, env, args, _ = load_model_and_env(
        metadata["checkpoint_path"], architecture=metadata["architecture"], device=device,
        seed=int(metadata["evaluation_seed"]), policy_dt=metadata.get("policy_dt"),
        env_overrides=dict(environment),
    )
    install_terminal_state_capture(env)
    steps = load_steps(result_dir)
    if sample_index < 0 or sample_index >= len(steps["true_state"]):
        env.close()
        raise IndexError("sample_index outside saved rollout.")
    if bool(steps["gate_crossing"][sample_index]) or bool(steps["terminated"][sample_index]) or bool(steps["truncated"][sample_index]):
        env.close()
        raise ValueError("Choose a saved sample away from a gate switch or terminal surface.")
    physical = np.asarray(steps["true_state"][sample_index], dtype=np.float64)
    hidden = np.asarray(steps["hidden_state"][sample_index], dtype=np.float64)
    target_index = int(steps["target_index_after"][sample_index])
    point = np.concatenate([physical, hidden])
    hidden_size = hidden.size

    def restore_and_step(combined: np.ndarray) -> np.ndarray:
        state_world = np.asarray(combined[:19], dtype=np.float64).copy()
        if hasattr(env, "sim_states"):
            env.target_gates[0] = target_index
            target = env.gate_pos[target_index % env.num_gates].astype(np.float64)
            state_world[0:3] -= target
            env.sim_states[0] = world_to_body_state(state_world).astype(np.float32)
            env.step_counts[0] = 0
            env.prev_derivs[0] = None
            env.state_hist[0] = 0.0
            env.action_hist[0] = 0.0
            env.update_states_gate()
            observation = env.states.copy()
        else:
            env.target_waypoints[0] = target_index
            target = env.waypoints[target_index % env.num_waypoints].astype(np.float64)
            state_world[0:3] -= target
            env.states[0] = world_to_body_state(state_world).astype(np.float32)
            env.step_counts[0] = 0
            env.prev_derivs[0] = None
            observation = env._observations()
        recurrent_state = None
        if hidden_size:
            state_shape = (1, 1, hidden_size)
            h = np.asarray(combined[19:], dtype=np.float32).reshape(state_shape)
            recurrent_state = (h, np.zeros_like(h))
        action, next_recurrent = model.predict(
            observation, state=recurrent_state, episode_start=np.zeros(1, dtype=bool), deterministic=True
        )
        _, _, dones, infos = env.step(action)
        info = infos[0]
        if bool(dones[0]) or bool(info.get("gate_passed", info.get("waypoint_reached", False))):
            raise ValueError("Finite-difference perturbation reached a discrete switch/terminal surface; choose another sample or epsilon.")
        next_physical, next_target = true_state_snapshot(env, terminal=False)
        if next_target != target_index:
            raise ValueError("Target index changed across local map.")
        next_hidden = np.asarray(next_recurrent[0]).reshape(-1) if next_recurrent is not None else np.empty(0)
        return np.concatenate([next_physical, next_hidden])

    analyses = []
    jacobians = []
    try:
        for epsilon in epsilon_values:
            if epsilon <= 0.0:
                raise ValueError("epsilon values must be positive.")
            jacobian, residual = central_difference_jacobian(restore_and_step, point, float(epsilon))
            jacobians.append(jacobian)
            analyses.append({"epsilon": float(epsilon), "forward_central_relative_residual": residual, **spectral_summary(jacobian)})
    finally:
        env.close()
    result = {
        "scope": "sample-local actual discrete closed-loop map [physical plant state, actuator state, external moment state, actor hidden]",
        "discrete_phase": {"target_index_held": target_index, "gate_switches_rejected": True},
        "sample_index": sample_index,
        "state_dimension": int(point.size),
        "coordinate_warning": "A single scalar epsilon is applied in mixed physical units; compare the reported epsilon sweep before interpretation.",
        "epsilon_analyses": analyses,
        "interpretation": "This is not a global nonlinear stability claim.",
    }
    np.savez_compressed(result_dir / "closed_loop_stability.npz", point=point, **{f"jacobian_{i}": value for i, value in enumerate(jacobians)})
    write_json(result_dir / "closed_loop_stability.json", result)
    return result
