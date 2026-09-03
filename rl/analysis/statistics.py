"""Statistical summaries that preserve the training-seed sampling unit."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable, Mapping

import numpy as np


def mean_confidence_interval(values: Iterable[float], confidence: float = 0.95) -> dict[str, float | int]:
    """Return a Student-t interval around a sample mean.

    A single observation has a defined mean but no estimable between-sample
    interval, represented by NaN bounds rather than a fabricated zero-width CI.
    """

    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    mean = float(np.mean(array))
    if n == 1:
        return {"n": 1, "mean": mean, "std": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    std = float(np.std(array, ddof=1))
    try:
        from scipy.stats import t

        critical = float(t.ppf(0.5 + confidence / 2.0, df=n - 1))
    except ImportError:  # pragma: no cover - exercised only on minimal installations
        # Exact two-sided 95% table keeps the default result Student-t based on
        # lightweight analysis hosts where SciPy is intentionally absent.
        t95 = (
            12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
            2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
            2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
        )
        if confidence == 0.95 and n - 1 <= len(t95):
            critical = t95[n - 2]
        else:
            critical = float(NormalDist().inv_cdf(0.5 + confidence / 2.0))
    half_width = critical * std / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
    }


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> dict[str, float | int]:
    """Wilson score interval for an episode-level Bernoulli proportion."""

    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Require 0 <= successes <= total and total > 0.")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "successes": int(successes),
        "total": int(total),
        "proportion": float(p),
        "ci_low": float(max(0.0, center - half)),
        "ci_high": float(min(1.0, center + half)),
    }


def aggregate_training_seeds(
    seed_summaries: Iterable[Mapping[str, float]],
    metrics: Iterable[str] | None = None,
) -> dict[str, object]:
    """Aggregate seed-level means without pooling their episodes.

    ``seed_summaries`` must contain one mapping per independently trained model.
    Any episode count/standard deviation in those mappings is retained only in
    ``individual_seeds``; the CI is computed across one value per training seed.
    """

    individual = [dict(summary) for summary in seed_summaries]
    if metrics is None:
        common = set.intersection(
            *(set(item) for item in individual)
        ) if individual else set()
        excluded = {"training_seed", "evaluation_seed", "seed", "episode_id"}
        metrics = sorted(
            key for key in common
            if key not in excluded
            if all(isinstance(item.get(key), (int, float, np.number)) for item in individual)
        )
    between = {}
    for metric in metrics:
        values = [float(item[metric]) for item in individual if metric in item]
        between[metric] = mean_confidence_interval(values)
    return {
        "sampling_unit": "independent_training_seed",
        "individual_seeds": individual,
        "between_training_seeds": between,
    }
