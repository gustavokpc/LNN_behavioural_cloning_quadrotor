"""Optional local and finite-history policy sensitivity analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import read_json, write_json
from .loading import load_model_and_env
from .rollout import load_steps


def _recurrent_state_tensors(policy, hidden: np.ndarray | None, reference):
    import torch

    shape = tuple(int(value) for value in policy.lstm_hidden_state_shape)
    shape = (shape[0], 1, shape[2])
    h = torch.zeros(shape, device=reference.device, dtype=reference.dtype)
    if hidden is not None and np.asarray(hidden).size:
        h = torch.as_tensor(hidden, device=reference.device, dtype=reference.dtype).reshape(shape)
    return (h, torch.zeros_like(h))


def _deterministic_action(policy, observation, recurrent_state=None, episode_start=None):
    import torch

    if hasattr(policy, "lstm_actor"):
        if recurrent_state is None:
            recurrent_state = _recurrent_state_tensors(policy, None, observation)
        if episode_start is None:
            episode_start = torch.zeros((observation.shape[0],), device=observation.device, dtype=observation.dtype)
        distribution, next_state = policy.get_distribution(observation, recurrent_state, episode_start)
    else:
        distribution = policy.get_distribution(observation)
        next_state = None
    action = distribution.get_actions(deterministic=True)
    low = torch.as_tensor(policy.action_space.low, device=action.device, dtype=action.dtype)
    high = torch.as_tensor(policy.action_space.high, device=action.device, dtype=action.dtype)
    return torch.clamp(action, min=low, max=high), next_state


def instantaneous_sensitivity(
    result_dir: str | Path,
    checkpoint: str | Path | None = None,
    *,
    architecture: str = "auto",
    device: str = "auto",
    samples: int = 64,
    event: str | None = None,
    phase: str = "all",
) -> dict[str, object]:
    """Compute E[|du_i/dx_j|] while holding incoming hidden state fixed."""

    import torch

    result_dir = Path(result_dir)
    metadata = read_json(result_dir / "metadata.json")
    checkpoint = checkpoint or metadata["checkpoint_path"]
    env_config = metadata.get("environment", {})
    overrides = {key: value for key, value in env_config.items() if key not in {"track"}}
    model, env, _, _ = load_model_and_env(
        checkpoint, architecture=architecture if architecture != "auto" else metadata["architecture"],
        device=device, seed=int(metadata["evaluation_seed"]), env_overrides=overrides,
        policy_dt=metadata.get("policy_dt"),
    )
    env.close()
    steps = load_steps(result_dir)
    if event:
        candidates = np.flatnonzero(np.asarray(steps[event], dtype=bool))
    else:
        candidates = np.arange(len(steps["observation"]))
    if phase != "all":
        state = steps["true_state"]
        if phase == "stabilized":
            phase_mask = (np.linalg.norm(state[:, 3:6], axis=1) < 0.3) & (np.linalg.norm(state[:, 9:12], axis=1) < 0.3)
        elif phase == "gate_approach":
            phase_mask = np.linalg.norm(steps["observation"][:, 0:3], axis=1) < 1.0
        elif phase == "turning":
            phase_mask = np.linalg.norm(state[:, 9:12], axis=1) >= 0.3
        elif phase == "recovery":
            onset = np.flatnonzero(steps["perturbation_onset"] | steps["observation_restoration"])
            phase_mask = np.zeros(len(state), dtype=bool)
            for index in onset:
                phase_mask[index:min(index + 100, len(state))] = steps["episode_id"][index:min(index + 100, len(state))] == steps["episode_id"][index]
        else:
            raise ValueError(f"Unknown phase: {phase}")
        candidates = candidates[phase_mask[candidates]]
    if not len(candidates):
        raise ValueError(f"No samples available for event {event!r}.")
    selected = candidates[np.linspace(0, len(candidates) - 1, min(samples, len(candidates)), dtype=int)]
    policy = model.policy
    policy.set_training_mode(False)
    matrices = []
    for index in selected:
        observation = torch.as_tensor(
            steps["policy_observation"][index:index + 1], dtype=torch.float32, device=policy.device
        ).detach().clone().requires_grad_(True)
        incoming_hidden = None
        if hasattr(policy, "lstm_actor"):
            same_episode = index > 0 and steps["episode_id"][index - 1] == steps["episode_id"][index]
            previous = steps["hidden_state"][index - 1] if same_episode and steps["hidden_state"].shape[1] else None
            incoming_hidden = _recurrent_state_tensors(policy, previous, observation)
        action, _ = _deterministic_action(policy, observation, incoming_hidden)
        rows = []
        for action_index in range(action.shape[1]):
            gradient = torch.autograd.grad(action[0, action_index], observation, retain_graph=True)[0]
            rows.append(gradient[0].detach().cpu().numpy())
        matrices.append(np.asarray(rows))
    matrices_array = np.asarray(matrices)
    matrix = np.mean(np.abs(matrices_array), axis=0)
    mean_jacobian = np.mean(matrices_array, axis=0)
    std_jacobian = np.std(matrices_array, axis=0)
    result = {
        "description": "instantaneous local action sensitivity with incoming recurrent hidden state held fixed",
        "action_coordinates": "policy action after deterministic distribution mean and action-bound clipping",
        "sample_indices": selected.astype(int).tolist(),
        "event_filter": event,
        "phase_filter": phase,
        "mean_absolute_jacobian": matrix.tolist(),
        "mean_jacobian": mean_jacobian.tolist(),
        "std_jacobian": std_jacobian.tolist(),
    }
    np.savez_compressed(
        result_dir / "instantaneous_sensitivity.npz", matrix=matrix, mean_jacobian=mean_jacobian,
        std_jacobian=std_jacobian, sample_indices=selected,
    )
    write_json(result_dir / "instantaneous_sensitivity.json", result)
    return result


def history_saliency(
    result_dir: str | Path,
    checkpoint: str | Path | None = None,
    *,
    architecture: str = "auto",
    device: str = "auto",
    history: int = 50,
    samples: int = 16,
) -> dict[str, object]:
    """Differentiate through finite recurrent unrolls without detaching hidden state."""

    import torch

    result_dir = Path(result_dir)
    metadata = read_json(result_dir / "metadata.json")
    checkpoint = checkpoint or metadata["checkpoint_path"]
    env_config = metadata.get("environment", {})
    overrides = {key: value for key, value in env_config.items() if key != "track"}
    model, env, _, _ = load_model_and_env(
        checkpoint, architecture=architecture if architecture != "auto" else metadata["architecture"],
        device=device, seed=int(metadata["evaluation_seed"]), env_overrides=overrides,
        policy_dt=metadata.get("policy_dt"),
    )
    env.close()
    policy = model.policy
    if not hasattr(policy, "lstm_actor"):
        raise ValueError("History saliency requires a recurrent policy; MLP is unsupported.")
    policy.set_training_mode(False)
    steps = load_steps(result_dir)
    episodes = steps["episode_id"]
    candidate_ends = []
    for episode in np.unique(episodes):
        indices = np.flatnonzero(episodes == episode)
        candidate_ends.extend(indices[history - 1:].tolist())
    if not candidate_ends:
        raise ValueError("No episode is long enough for the requested history.")
    selected = np.asarray(candidate_ends)[np.linspace(0, len(candidate_ends) - 1, min(samples, len(candidate_ends)), dtype=int)]
    saliencies = []
    for end in selected:
        start = int(end - history + 1)
        observations = [
            torch.as_tensor(steps["policy_observation"][idx:idx + 1], dtype=torch.float32, device=policy.device)
            .detach().clone().requires_grad_(True)
            for idx in range(start, int(end) + 1)
        ]
        preceding_same_episode = start > 0 and episodes[start - 1] == episodes[start]
        previous_hidden = steps["hidden_state"][start - 1] if preceding_same_episode and steps["hidden_state"].shape[1] else None
        state = _recurrent_state_tensors(policy, previous_hidden, observations[0])
        action = None
        for offset, observation in enumerate(observations):
            episode_start = torch.as_tensor(
                [float(offset == 0 and not preceding_same_episode)], device=policy.device, dtype=observation.dtype
            )
            action, state = _deterministic_action(policy, observation, state, episode_start)
        per_action = []
        for action_index in range(action.shape[1]):
            gradients = torch.autograd.grad(
                action[0, action_index], observations, retain_graph=action_index + 1 < action.shape[1], allow_unused=False
            )
            per_action.append(np.asarray([gradient[0].detach().cpu().numpy() for gradient in gradients]))
        # (action, chronological time, observation) -> lag 0 first
        saliencies.append(np.abs(np.asarray(per_action))[:, ::-1, :])
    mean_by_action = np.mean(np.asarray(saliencies), axis=0)
    matrix = np.mean(mean_by_action, axis=0)
    lag_strength = np.sum(matrix, axis=1)
    cumulative = np.cumsum(lag_strength)
    horizon = int(np.searchsorted(cumulative, 0.95 * cumulative[-1])) if cumulative.size and cumulative[-1] > 0 else None
    result = {
        "description": "S(k,j)=mean absolute gradient through finite recurrent unroll; lag 0 is current observation",
        "history_steps": history,
        "history_seconds": history * float(metadata["dt"]),
        "sample_end_indices": selected.astype(int).tolist(),
        "lag_observation_saliency": matrix.tolist(),
        "per_action_lag_observation_saliency": mean_by_action.tolist(),
        "effective_memory_horizon_95pct_steps": horizon,
        "effective_memory_horizon_95pct_seconds": horizon * float(metadata["dt"]) if horizon is not None else None,
    }
    np.savez_compressed(result_dir / "history_saliency.npz", matrix=matrix, per_action=mean_by_action, sample_indices=selected)
    write_json(result_dir / "history_saliency.json", result)
    return result


def virtual_control_gain(jacobian: np.ndarray) -> dict[str, object]:
    """Transform four normalized motors into an explicitly approximate mixer basis."""

    mixer = np.asarray([
        [1.0, 1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0, 1.0],
    ])
    return {
        "labels": ["total_command", "roll_mix", "pitch_mix", "yaw_mix"],
        "gain": (mixer @ np.asarray(jacobian)).tolist(),
        "warning": "Normalized-command mixer basis only; it is not a calibrated force/moment Jacobian.",
    }
