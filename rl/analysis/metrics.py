"""Per-episode task and control metrics."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

import numpy as np

from .statistics import wilson_interval


def _rms_rows(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    if values.ndim == 1:
        values = values[:, None]
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def action_metrics(actions: np.ndarray, prefix: str = "action") -> dict[str, object]:
    """Compute vector and per-motor magnitude/rate/second-difference RMS."""

    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError("actions must have shape (time, motors).")
    delta = np.diff(actions, axis=0)
    jerk = np.diff(actions, n=2, axis=0)

    def per_motor(values: np.ndarray) -> list[float]:
        if values.shape[0] == 0:
            return [float("nan")] * actions.shape[1]
        return np.sqrt(np.mean(values * values, axis=0)).astype(float).tolist()

    return {
        f"{prefix}_rms": _rms_rows(actions),
        f"{prefix}_delta_rms": _rms_rows(delta),
        f"{prefix}_jerk_rms": _rms_rows(jerk),
        f"{prefix}_per_motor_rms": per_motor(actions),
        f"{prefix}_per_motor_delta_rms": per_motor(delta),
        f"{prefix}_per_motor_jerk_rms": per_motor(jerk),
    }


def termination_reasons(final_info: Mapping[str, object], success: bool = False) -> list[str]:
    """Return every environment-reported active terminal cause, without guessing priority."""

    keys = (
        "gate_collision",
        "ground_collision",
        "out_of_bounds",
        "attitude_crash",
        "unstable_termination",
    )
    active = [key for key in keys if bool(final_info.get(key, False))]
    if bool(final_info.get("TimeLimit.truncated", False)):
        active.append("timeout")
    if active:
        return active
    return ["success"] if success else ["other"]


def compute_episode_metrics(
    *,
    rewards: np.ndarray,
    actions: np.ndarray,
    command_actions: np.ndarray | None,
    rpm_commands: np.ndarray | None,
    true_states: np.ndarray,
    gate_events: np.ndarray,
    dt: float,
    final_info: Mapping[str, object],
    num_gates: int = 8,
) -> dict[str, object]:
    """Compute metrics for one complete episode before any aggregation."""

    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    true_states = np.asarray(true_states, dtype=np.float64)
    gate_events = np.asarray(gate_events, dtype=bool).reshape(-1)
    gates = int(np.count_nonzero(gate_events))
    cycle_indices = np.flatnonzero(np.cumsum(gate_events) >= num_gates)
    full_cycle = bool(cycle_indices.size)
    rates = true_states[:, 9:12] if true_states.ndim == 2 and true_states.shape[1] >= 12 else np.empty((0, 3))
    reasons = termination_reasons(final_info, success=full_cycle or bool(final_info.get("is_success", False)))
    result: dict[str, object] = {
        "total_reward": float(np.sum(rewards)),
        "gates_crossed": gates,
        "full_cycle_success": full_cycle,
        "success": bool(full_cycle or final_info.get("is_success", False)),
        "failure": not bool(full_cycle or final_info.get("is_success", False)),
        "first_cycle_time_s": float((cycle_indices[0] + 1) * dt) if full_cycle else float("nan"),
        "episode_duration_s": float(len(rewards) * dt),
        "episode_steps": int(len(rewards)),
        "max_abs_angular_rate": float(np.max(np.abs(rates))) if rates.size else float("nan"),
        "max_angular_rate_norm": float(np.max(np.linalg.norm(rates, axis=1))) if rates.size else float("nan"),
        "termination_reason": "+".join(reasons),
        "termination_reasons": reasons,
    }
    result.update(action_metrics(actions, "policy_action"))
    if command_actions is not None:
        result.update(action_metrics(command_actions, "motor_command_01"))
    if rpm_commands is not None:
        result.update(action_metrics(rpm_commands, "motor_command_rpm"))
    return result


def aggregate_episodes(episodes: Iterable[Mapping[str, object]]) -> dict[str, object]:
    episodes = [dict(episode) for episode in episodes]
    if not episodes:
        raise ValueError("At least one episode is required.")
    scalar_metrics: dict[str, dict[str, float | int]] = {}
    ignored = {"episode_id", "evaluation_seed", "termination_reason", "termination_reasons"}
    keys = set().union(*(episode.keys() for episode in episodes)) - ignored
    for key in sorted(keys):
        values = [episode[key] for episode in episodes if key in episode]
        if values and all(isinstance(value, (bool, int, float, np.number)) for value in values):
            array = np.asarray(values, dtype=np.float64)
            finite = array[np.isfinite(array)]
            scalar_metrics[key] = {
                "n": int(finite.size),
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "std": float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
            }
    successes = sum(bool(episode.get("success", False)) for episode in episodes)
    reason_counts: Counter[str] = Counter()
    for episode in episodes:
        reason_counts.update(episode.get("termination_reasons", [episode.get("termination_reason", "other")]))
    result = {
        "num_episodes": len(episodes),
        "metrics": scalar_metrics,
        "success_wilson_95": wilson_interval(successes, len(episodes)),
        "failure_breakdown": {
            reason: {"count": count, "rate": count / len(episodes)}
            for reason, count in sorted(reason_counts.items())
        },
        "note": "Episode variability is summarized here; training-seed variability must use seed aggregation.",
    }
    if all("full_cycle_success" in episode for episode in episodes):
        cycle_successes = sum(bool(episode["full_cycle_success"]) for episode in episodes)
        result["full_cycle_success_wilson_95"] = wilson_interval(cycle_successes, len(episodes))
    return result


def degradation(stressed: float, nominal: float, higher_is_better: bool = True) -> dict[str, float]:
    """Absolute change, relative change, and signed percentage degradation."""

    absolute = float(stressed - nominal)
    relative = absolute / abs(nominal) if nominal != 0 else float("nan")
    percentage_degradation = (-relative if higher_is_better else relative) * 100.0
    return {
        "absolute_difference": absolute,
        "relative_difference": float(relative),
        "percentage_degradation": float(percentage_degradation),
    }
