"""Checkpoint-based learning/sample-efficiency evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .io import flatten_summary_means, read_json, write_json, write_rows_csv
from .loading import checkpoint_step, recover_checkpoint_step
from .rollout import RolloutConfig, collect_rollouts


def discover_checkpoints(reference: str | Path) -> list[Path]:
    """Find same-run checkpoints only when the step suffix is recoverable."""

    reference = Path(reference).resolve()
    base = reference.stem
    if checkpoint_step(reference) is not None:
        import re

        base = re.sub(r"_\d+_steps$", "", base)
    candidates = sorted(reference.parent.glob(f"{base}_*_steps.zip"), key=lambda path: checkpoint_step(path) or -1)
    if reference.exists() and recover_checkpoint_step(reference) is not None and reference not in candidates:
        candidates.append(reference)
    return sorted(set(candidates), key=lambda path: recover_checkpoint_step(path) or -1)


def evaluate_learning_curve(
    checkpoints: Iterable[str | Path],
    output_dir: str | Path,
    *,
    architecture: str = "auto",
    episodes: int = 5,
    seed: int = 0,
    device: str = "auto",
    success_threshold: float = 0.8,
    env_overrides: dict | None = None,
) -> dict[str, object]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    missing_steps = []
    for checkpoint in checkpoints:
        step = recover_checkpoint_step(checkpoint)
        if step is None:
            missing_steps.append(str(checkpoint))
            continue
        result_dir = output / f"checkpoint_{step}"
        collect_rollouts(RolloutConfig(
            checkpoint=str(checkpoint), output_dir=str(result_dir), architecture=architecture,
            label=f"learning_curve_step_{step}", episodes=episodes, seed=seed, device=device,
            env_overrides=dict(env_overrides or {}),
        ))
        summary = read_json(result_dir / "summary.json")
        means = flatten_summary_means(summary)
        rows.append({"environment_steps": step, "checkpoint": str(Path(checkpoint).resolve()), **means})
    rows.sort(key=lambda row: row["environment_steps"])
    steps = np.asarray([row["environment_steps"] for row in rows], dtype=np.float64)
    first_success = next((row["environment_steps"] for row in rows if row.get("full_cycle_success", row.get("success", 0.0)) > 0.0), None)
    threshold_step = next((row["environment_steps"] for row in rows if row.get("full_cycle_success", row.get("success", 0.0)) >= success_threshold), None)
    auc = {}
    for metric in ("total_reward", "gates_crossed", "full_cycle_success", "success"):
        values = np.asarray([row.get(metric, np.nan) for row in rows], dtype=np.float64)
        valid = np.isfinite(values)
        auc[metric] = float(np.trapz(values[valid], steps[valid])) if valid.sum() >= 2 else None
    result = {
        "checkpoint_results": rows,
        "skipped_checkpoints_without_recoverable_step": missing_steps,
        "steps_to_first_successful_checkpoint": first_success,
        "steps_to_success_rate_threshold": threshold_step,
        "success_rate_threshold": success_threshold,
        "area_under_learning_curve": auc,
        "auc_note": "Raw trapezoidal AUC in metric * environment-step units.",
    }
    write_rows_csv(output / "learning_curve.csv", rows)
    write_json(output / "learning_curve.json", result)
    return result
