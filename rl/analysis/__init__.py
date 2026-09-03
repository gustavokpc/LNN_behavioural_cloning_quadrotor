"""Reproducible, evaluation-only analysis tools for the RL policies.

The package deliberately depends on the existing training/environment modules for
model construction and simulation.  Pure numerical utilities remain importable
without Stable-Baselines3 so saved results can be analysed on lightweight hosts.
"""

from .metrics import action_metrics, compute_episode_metrics, degradation
from .statistics import aggregate_training_seeds, mean_confidence_interval, wilson_interval

__all__ = [
    "action_metrics",
    "aggregate_training_seeds",
    "compute_episode_metrics",
    "degradation",
    "mean_confidence_interval",
    "wilson_interval",
]
