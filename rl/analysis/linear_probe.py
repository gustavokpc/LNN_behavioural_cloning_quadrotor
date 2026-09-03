"""Episode-split ridge probes for recurrent representations."""

from __future__ import annotations

from typing import Iterable

import numpy as np


TRUE_STATE_GROUPS = {
    "position": (0, 1, 2),
    "velocity": (3, 4, 5),
    "attitude": (6, 7, 8),
    "angular_rates": (9, 10, 11),
}
TRUE_STATE_LABELS = ("x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r")


def split_episode_ids(
    episode_ids: np.ndarray,
    test_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split complete episodes, never transitions within an episode."""

    unique = np.unique(np.asarray(episode_ids))
    if unique.size < 2:
        raise ValueError("Linear probing requires at least two complete episodes.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_test = min(max(1, int(round(unique.size * test_fraction))), unique.size - 1)
    return np.sort(shuffled[n_test:]), np.sort(shuffled[:n_test])


def r2_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = np.sum((y_true - y_pred) ** 2, axis=0)
    total = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    return np.where(total > 0.0, 1.0 - residual / total, np.nan)


def fit_episode_ridge_probe(
    hidden: np.ndarray,
    true_state: np.ndarray,
    episode_ids: np.ndarray,
    *,
    alpha: float = 1.0,
    test_fraction: float = 0.25,
    seed: int = 0,
) -> dict[str, object]:
    """Fit standardized ridge regression using complete-episode train/test sets."""

    hidden = np.asarray(hidden, dtype=np.float64)
    targets = np.asarray(true_state, dtype=np.float64)[:, :12]
    episode_ids = np.asarray(episode_ids)
    if hidden.ndim != 2 or targets.ndim != 2 or len(hidden) != len(targets) or len(hidden) != len(episode_ids):
        raise ValueError("hidden, true_state, and episode_ids must share their first dimension.")
    train_ids, test_ids = split_episode_ids(episode_ids, test_fraction=test_fraction, seed=seed)
    train = np.isin(episode_ids, train_ids)
    test = np.isin(episode_ids, test_ids)
    x_mean = hidden[train].mean(axis=0)
    x_std = hidden[train].std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    y_mean = targets[train].mean(axis=0)
    y_std = targets[train].std(axis=0)
    y_std[y_std < 1e-12] = 1.0
    x_train = (hidden[train] - x_mean) / x_std
    x_test = (hidden[test] - x_mean) / x_std
    y_train = (targets[train] - y_mean) / y_std
    design = np.column_stack([x_train, np.ones(len(x_train))])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    pred_standard = np.column_stack([x_test, np.ones(len(x_test))]) @ weights
    prediction = pred_standard * y_std + y_mean
    scores = r2_per_target(targets[test], prediction)
    grouped = {
        group: float(np.nanmean(scores[list(indices)]))
        for group, indices in TRUE_STATE_GROUPS.items()
    }
    return {
        "alpha": float(alpha),
        "train_episode_ids": train_ids.astype(int).tolist(),
        "test_episode_ids": test_ids.astype(int).tolist(),
        "train_samples": int(train.sum()),
        "test_samples": int(test.sum()),
        "r2": {label: float(value) for label, value in zip(TRUE_STATE_LABELS, scores)},
        "grouped_mean_r2": grouped,
    }


def compare_partial_observability(
    partial: dict[str, float],
    full: dict[str, float],
    metrics: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Architecture-matched full/partial comparison."""

    from .metrics import degradation

    return {
        metric: degradation(float(partial[metric]), float(full[metric]), higher_is_better=True)
        for metric in metrics
    }
