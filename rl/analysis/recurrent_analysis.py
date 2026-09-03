"""Saved recurrent-state summaries and physical-time event alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .io import read_json, write_json
from .rollout import load_steps


def summarize_recurrent(result_dir: str | Path) -> dict[str, object]:
    steps = load_steps(result_dir)
    hidden = np.asarray(steps.get("hidden_state", np.empty((0, 0))), dtype=np.float64)
    if hidden.ndim != 2 or hidden.shape[1] == 0:
        return {"supported": False, "reason": "No hidden state in steps.npz; evaluate with --record-recurrent."}
    norm = np.linalg.norm(hidden, axis=1)
    extrema = np.max(np.abs(hidden), axis=0)
    saturation_threshold = 0.95
    result: dict[str, object] = {
        "supported": True,
        "samples": int(hidden.shape[0]),
        "neurons": int(hidden.shape[1]),
        "hidden_norm": {
            "mean": float(np.mean(norm)),
            "std": float(np.std(norm, ddof=1)) if len(norm) > 1 else 0.0,
            "min": float(np.min(norm)),
            "max": float(np.max(norm)),
        },
        "per_neuron_mean": np.mean(hidden, axis=0).tolist(),
        "per_neuron_std": np.std(hidden, axis=0).tolist(),
        "per_neuron_max_abs": extrema.tolist(),
        "fraction_abs_ge_0_95": np.mean(np.abs(hidden) >= saturation_threshold, axis=0).tolist(),
        "saturation_note": "|h|>=0.95 is an occupancy indicator, not a universal cell saturation proof.",
    }
    for name in ("cfc_gate", "cfc_tau", "ltc_tau"):
        if name in steps:
            values = np.asarray(steps[name], dtype=np.float64)
            temporal_mean = np.mean(values, axis=0)
            temporal_std = np.std(values, axis=0)
            result[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean_temporal_coefficient_of_variation": float(
                    np.nanmean(temporal_std / np.maximum(np.abs(temporal_mean), 1e-12))
                ),
            }
    return result


def align_events(
    result_dir: str | Path,
    event: str,
    *,
    pre_s: float = 0.5,
    post_s: float = 1.0,
    neuron_indices: Iterable[int] | None = None,
    outcome: str = "all",
    gate_approach_distance: float = 1.0,
) -> dict[str, np.ndarray]:
    """Align hidden/action/state traces without crossing episode boundaries."""

    steps = load_steps(result_dir)
    metadata = read_json(Path(result_dir) / "metadata.json")
    dt = float(metadata["dt"])
    event_aliases = {
        "gate_crossing": "gate_crossing",
        "dropout_onset": "dropout_onset",
        "observation_restoration": "observation_restoration",
        "perturbation_onset": "perturbation_onset",
        "failure": "terminated",
        "episode_start": "episode_start",
        "gate_approach": "gate_approach",
    }
    if event not in event_aliases:
        raise ValueError(f"Unknown event: {event}")
    if event == "episode_start":
        flags = np.r_[True, np.diff(steps["episode_id"]) != 0]
    elif event == "gate_approach":
        distance = np.linalg.norm(steps["observation"][:, 0:3], axis=1)
        near = distance <= gate_approach_distance
        flags = near & np.r_[True, ~near[:-1]]
        flags &= np.r_[True, np.diff(steps["episode_id"]) == 0]
    else:
        flags = np.asarray(steps[event_aliases[event]], dtype=bool)
    event_indices = np.flatnonzero(flags)
    if outcome not in {"all", "success", "failure"}:
        raise ValueError("outcome must be all, success, or failure.")
    if outcome != "all":
        successful_episodes = set()
        for episode_id in np.unique(steps["episode_id"]):
            mask = steps["episode_id"] == episode_id
            terminal_reasons = set(steps["termination_reason"][mask].tolist())
            gate_count = int(np.count_nonzero(steps["gate_crossing"][mask]))
            if gate_count >= 8 or "success" in terminal_reasons:
                successful_episodes.add(int(episode_id))
        wanted_success = outcome == "success"
        event_indices = np.asarray([
            index for index in event_indices
            if (int(steps["episode_id"][index]) in successful_episodes) == wanted_success
        ], dtype=int)
    before = int(round(pre_s / dt))
    after = int(round(post_s / dt))
    offsets = np.arange(-before, after + 1)
    episode_ids = steps["episode_id"]
    signals: dict[str, np.ndarray] = {
        "time_s": offsets.astype(np.float64) * dt,
        "event_indices": event_indices,
    }
    source_signals = {
        "hidden_norm": np.linalg.norm(steps["hidden_state"], axis=1) if steps["hidden_state"].shape[1] else None,
        "action_magnitude": np.linalg.norm(steps["policy_action"], axis=1),
        "angular_rate_norm": np.linalg.norm(steps["true_state"][:, 9:12], axis=1),
        "speed": np.linalg.norm(steps["true_state"][:, 3:6], axis=1),
    }
    action_delta = np.linalg.norm(np.diff(steps["policy_action"], axis=0, prepend=steps["policy_action"][:1]), axis=1)
    source_signals["action_delta_magnitude"] = action_delta
    if neuron_indices is not None and steps["hidden_state"].shape[1]:
        for neuron in neuron_indices:
            source_signals[f"hidden_neuron_{neuron}"] = steps["hidden_state"][:, int(neuron)]
    for name in ("cfc_gate", "cfc_tau", "ltc_tau"):
        if name in steps:
            source_signals[f"{name}_mean"] = np.mean(steps[name], axis=1)
    for name, values in source_signals.items():
        if values is None:
            continue
        aligned = np.full((len(event_indices), len(offsets)), np.nan, dtype=np.float64)
        for row, center in enumerate(event_indices):
            indices = center + offsets
            valid = (indices >= 0) & (indices < len(values))
            valid &= episode_ids[np.clip(indices, 0, len(values) - 1)] == episode_ids[center]
            aligned[row, valid] = values[indices[valid]]
        signals[name] = aligned
        signals[f"{name}_mean"] = np.nanmean(aligned, axis=0) if len(aligned) else np.full(len(offsets), np.nan)
        signals[f"{name}_std"] = np.nanstd(aligned, axis=0) if len(aligned) else np.full(len(offsets), np.nan)
    return signals


def save_recurrent_analysis(
    result_dir: str | Path,
    *,
    event: str = "gate_crossing",
    pre_s: float = 0.5,
    post_s: float = 1.0,
    neuron_indices: Iterable[int] | None = None,
    outcome: str = "all",
) -> dict[str, object]:
    result_dir = Path(result_dir)
    summary = summarize_recurrent(result_dir)
    write_json(result_dir / "recurrent_summary.json", summary)
    if summary.get("supported"):
        aligned = align_events(
            result_dir, event, pre_s=pre_s, post_s=post_s,
            neuron_indices=neuron_indices, outcome=outcome,
        )
        np.savez_compressed(result_dir / f"event_{event}.npz", **aligned)
    return summary
