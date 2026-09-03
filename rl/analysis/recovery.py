"""Practical, explicitly defined perturbation/dropout recovery metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import read_json, write_json
from .rollout import load_steps


def recovery_metrics(
    result_dir: str | Path,
    *,
    event: str = "perturbation_onset",
    position_band_m: float = 0.20,
    rate_band_rad_s: float = 0.20,
    dwell_s: float = 0.25,
) -> dict[str, object]:
    """Measure return to the pre-event position/rate neighborhood.

    Settling means remaining simultaneously within both bands for ``dwell_s``.
    This is a trajectory-recovery diagnostic, not a Lyapunov certificate.
    """

    path = Path(result_dir)
    steps = load_steps(path)
    dt = float(read_json(path / "metadata.json")["dt"])
    flags = np.asarray(steps[event], dtype=bool)
    dwell_steps = max(1, int(round(dwell_s / dt)))
    rows = []
    for index in np.flatnonzero(flags):
        episode = steps["episode_id"][index]
        episode_end = index
        while episode_end + 1 < len(flags) and steps["episode_id"][episode_end + 1] == episode:
            episode_end += 1
        reference_index = max(index - 1, 0)
        state = steps["true_state"][index:episode_end + 1]
        reference = steps["true_state"][reference_index]
        position_error = np.linalg.norm(state[:, 0:3] - reference[0:3], axis=1)
        rate_error = np.linalg.norm(state[:, 9:12] - reference[9:12], axis=1)
        inside = (position_error <= position_band_m) & (rate_error <= rate_band_rad_s)
        settled_at = None
        for offset in range(max(0, len(inside) - dwell_steps + 1)):
            if np.all(inside[offset:offset + dwell_steps]):
                settled_at = offset
                break
        action = steps["motor_command_01"][index:episode_end + 1]
        delta = np.diff(action, axis=0)
        rows.append({
            "episode_id": int(episode),
            "peak_position_deviation_m": float(np.max(position_error)),
            "peak_rate_deviation_rad_s": float(np.max(rate_error)),
            "position_overshoot_m": float(max(0.0, np.max(position_error) - position_error[0])),
            "recovery_time_s": float(settled_at * dt) if settled_at is not None else None,
            "settled": settled_at is not None,
            "control_effort_rms": float(np.sqrt(np.mean(np.sum(action * action, axis=1)))),
            "control_delta_rms": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))) if len(delta) else None,
            "position_error_turning_points": int(np.count_nonzero(np.diff(np.sign(np.diff(position_error))))),
        })
    result = {
        "definition": {
            "reference": "true physical state immediately before event",
            "position_band_m": position_band_m,
            "angular_rate_band_rad_s": rate_band_rad_s,
            "minimum_dwell_s": dwell_s,
        },
        "events": rows,
    }
    write_json(path / f"recovery_{event}.json", result)
    return result
