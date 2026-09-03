"""Plots generated entirely from saved result files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_metric_comparison(result_dirs: Iterable[str | Path], metric: str, output: str | Path) -> Path:
    labels, means, errors = [], [], []
    for directory in result_dirs:
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        stats = summary["metrics"][metric]
        labels.append(metadata.get("experiment_label", directory.name))
        means.append(stats["mean"])
        errors.append(stats["std"] or 0.0)
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(max(5.0, len(labels) * 1.25), 3.8))
    axis.bar(labels, means, yerr=errors, capsize=4)
    axis.set_ylabel(metric.replace("_", " "))
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_failure_breakdown(result_dir: str | Path, output: str | Path) -> Path:
    summary = json.loads((Path(result_dir) / "summary.json").read_text())
    breakdown = summary["failure_breakdown"]
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(6.0, 3.8))
    axis.bar(list(breakdown), [item["rate"] for item in breakdown.values()])
    axis.set_ylabel("episode rate")
    axis.tick_params(axis="x", rotation=30)
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_learning_curve(curve_json: str | Path, metric: str, output: str | Path) -> Path:
    data = json.loads(Path(curve_json).read_text())
    rows = data["checkpoint_results"]
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(5.5, 3.8))
    axis.plot([row["environment_steps"] for row in rows], [row.get(metric, np.nan) for row in rows], marker="o")
    axis.set_xlabel("training environment steps")
    axis.set_ylabel(metric.replace("_", " "))
    axis.grid(alpha=0.25)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_robustness(rows: list[dict], metric: str, output: str | Path) -> Path:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(5.5, 3.8))
    axis.plot([row["stress_value"] for row in rows], [row[metric] for row in rows], marker="o")
    axis.set_xlabel("stress level")
    axis.set_ylabel(metric.replace("_", " "))
    axis.grid(alpha=0.25)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_probe_r2(probe: dict, output: str | Path) -> Path:
    values = probe["r2"]
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7.0, 3.8))
    axis.bar(list(values), list(values.values()))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("test R²")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_heatmap(matrix: np.ndarray, xlabels: list[str], ylabels: list[str], output: str | Path, title: str) -> Path:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(max(5.0, len(xlabels) * 0.35), max(3.0, len(ylabels) * 0.45)))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(xlabels)), xlabels, rotation=60, ha="right")
    axis.set_yticks(np.arange(len(ylabels)), ylabels)
    axis.set_title(title)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_event_traces(event_npz: str | Path, output: str | Path) -> Path:
    with np.load(event_npz) as data:
        time_s = data["time_s"]
        names = [
            name[:-5] for name in data.files
            if name.endswith("_mean") and data[name].ndim == 1 and data[name].shape == time_s.shape
        ]
        plt = _pyplot()
        fig, axes = plt.subplots(len(names), 1, figsize=(6.0, max(2.5, 2.1 * len(names))), sharex=True)
        axes = np.atleast_1d(axes)
        for axis, name in zip(axes, names):
            axis.plot(time_s, data[f"{name}_mean"])
            if f"{name}_std" in data:
                mean, std = data[f"{name}_mean"], data[f"{name}_std"]
                axis.fill_between(time_s, mean - std, mean + std, alpha=0.2)
            axis.set_ylabel(name.replace("_", " "))
            axis.axvline(0.0, color="black", linewidth=0.8)
            axis.grid(alpha=0.2)
        axes[-1].set_xlabel("time from event [s]")
        fig.tight_layout()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        plt.close(fig)
    return output


def plot_latency_comparison(efficiency_jsons: Iterable[str | Path], output: str | Path) -> Path:
    paths = [Path(path) for path in efficiency_jsons]
    rows = [json.loads(path.read_text()) for path in paths]
    labels = [row.get("architecture", path.stem) for row, path in zip(rows, paths)]
    mean_ms = [1000.0 * row["mean_latency_s"] for row in rows]
    p95_ms = [1000.0 * row["p95_latency_s"] for row in rows]
    plt = _pyplot()
    positions = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(max(5.0, 1.3 * len(rows)), 3.8))
    axis.bar(positions - 0.18, mean_ms, 0.36, label="mean")
    axis.bar(positions + 0.18, p95_ms, 0.36, label="p95")
    axis.set_xticks(positions, labels, rotation=25)
    axis.set_ylabel("policy-only latency [ms]")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_recovery_trace(result_dir: str | Path, event: str, output: str | Path) -> Path:
    result_dir = Path(result_dir)
    with np.load(result_dir / "steps.npz") as steps:
        flags = np.asarray(steps[event], dtype=bool)
        indices = np.flatnonzero(flags)
        if not len(indices):
            raise ValueError(f"No {event} event in saved steps.")
        index = int(indices[0])
        episode = steps["episode_id"][index]
        end = index
        while end + 1 < len(flags) and steps["episode_id"][end + 1] == episode:
            end += 1
        start = max(0, index - 1)
        state = steps["true_state"][start:end + 1]
        action = steps["motor_command_01"][start:end + 1]
        time_s = steps["time_s"][start:end + 1] - steps["time_s"][index]
        position_error = np.linalg.norm(state[:, :3] - state[0, :3], axis=1)
        rate_error = np.linalg.norm(state[:, 9:12] - state[0, 9:12], axis=1)
    plt = _pyplot()
    fig, axes = plt.subplots(3, 1, figsize=(6.0, 6.5), sharex=True)
    axes[0].plot(time_s, position_error)
    axes[0].set_ylabel("position deviation [m]")
    axes[1].plot(time_s, rate_error)
    axes[1].set_ylabel("rate deviation [rad/s]")
    axes[2].plot(time_s, action)
    axes[2].set_ylabel("motor command")
    axes[2].set_xlabel("time from event [s]")
    for axis in axes:
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_stability_summary(stability_json: str | Path, output: str | Path) -> Path:
    data = json.loads(Path(stability_json).read_text())
    analyses = data.get("epsilon_analyses", [data])
    labels = [str(item.get("epsilon", "autograd")) for item in analyses]
    radii = [item["spectral_radius"] for item in analyses]
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(5.0, 3.6))
    axis.bar(labels, radii)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=0.9, label="unit radius")
    axis.set_xlabel("finite-difference epsilon")
    axis.set_ylabel("local spectral radius")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
