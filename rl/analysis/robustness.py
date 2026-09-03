"""Saved-result robustness calculations and sweep orchestration helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .metrics import degradation


HIGHER_IS_BETTER = {
    "success": True,
    "success_rate": True,
    "full_cycle_success": True,
    "gates_crossed": True,
    "total_reward": True,
    "failure": False,
    "failure_rate": False,
    "policy_action_rms": False,
    "policy_action_delta_rms": False,
    "policy_action_jerk_rms": False,
}


def robustness_degradation(
    nominal: dict[str, float],
    stressed: dict[str, float],
    metrics: Iterable[str],
) -> dict[str, dict[str, float]]:
    return {
        metric: degradation(
            float(stressed[metric]),
            float(nominal[metric]),
            higher_is_better=HIGHER_IS_BETTER.get(metric, True),
        )
        for metric in metrics
    }


def normalized_robustness_auc(levels: Iterable[float], values: Iterable[float]) -> float:
    levels = np.asarray(list(levels), dtype=np.float64)
    values = np.asarray(list(values), dtype=np.float64)
    if levels.size < 2 or levels.shape != values.shape:
        return float("nan")
    order = np.argsort(levels)
    levels, values = levels[order], values[order]
    span = levels[-1] - levels[0]
    nominal = values[0]
    if span == 0.0 or nominal == 0.0:
        return float("nan")
    return float(np.trapz(values / abs(nominal), levels) / span)


def first_degradation_level(
    levels: Iterable[float],
    values: Iterable[float],
    threshold_percent: float,
    higher_is_better: bool = True,
) -> float | None:
    levels = list(levels)
    values = list(values)
    if not values:
        return None
    nominal = float(values[0])
    for level, value in zip(levels[1:], values[1:]):
        result = degradation(float(value), nominal, higher_is_better=higher_is_better)
        if result["percentage_degradation"] >= threshold_percent:
            return float(level)
    return None
