#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT.parent
sys.path.insert(0, str(REPO_ROOT))

from LNN_behavioural_cloning_quadrotor.utils.config import load_yaml
from LNN_behavioural_cloning_quadrotor.utils.data import expand_feature_labels
from LNN_behavioural_cloning_quadrotor.utils.dynamics_models import set_dynamics_model
from LNN_behavioural_cloning_quadrotor.utils.quadrotor_sim import (
    build_input_vector,
    build_lightning_model,
    integrate_state,
    make_model_observation,
    state_from_input_features,
)

BEBOP1_TRAJ = 245
BEBOP2_TRAJ = 208
HORIZON_SECONDS = 4.0
OMEGA_MIN = 5000.0
OMEGA_MAX = 10000.0

OUT_DIR = PROJECT / "datasets" / "diagnostics" / "matched_test_original_245_bebop2_corrected_208"
BEBOP1_DATASET = PROJECT / "datasets" / "hover_dataset_test.npz"
BEBOP2_DATASET = PROJECT / "datasets" / "hover_dataset_test_bebop2_corrected.npz"

BEBOP1_MODEL = "new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.ckpt"
BEBOP1_CONFIG = PROJECT / "configs" / "new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.yaml"
BEBOP1_NORM = "LNN_behavioural_cloning_quadrotor.utils.normalization_limits"

BEBOP2_MODEL = "NOVA_VERSAOZE_BEBP2_conv_cfc_default_n64_bebop2_epoch=19_val_loss=0.000098.ckpt"
BEBOP2_CONFIG = PROJECT / "configs" / "NOVA_VERSAOZE_BEBP2_conv_cfc_default_n64_bebop2_epoch=19_val_loss=0.000098.yaml"
BEBOP2_NORM = "LNN_behavioural_cloning_quadrotor.utils.normalization_limits_bebop2_new"


def _load_raw_trajectory(dataset_path: Path, index: int, input_labels: list[str]) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    with np.load(dataset_path) as data:
        columns: list[np.ndarray] = []
        expanded: list[str] = []
        for label in input_labels:
            if label in {"t", "dt"}:
                continue
            if label == "omega":
                omega = np.asarray(data["omega"][index], dtype=np.float64)
                columns.extend([omega[:, motor] for motor in range(4)])
                expanded.extend([f"omega{motor}" for motor in range(1, 5)])
            elif label in {"Mx_ext", "My_ext", "Mz_ext"}:
                value = float(data[label][index])
                length = int(data["dx"].shape[1])
                columns.append(np.full(length, value, dtype=np.float64))
                expanded.append(label)
            else:
                columns.append(np.asarray(data[label][index], dtype=np.float64))
                expanded.append(label)
        actions = np.asarray(data["u"][index], dtype=np.float64)
        # Match utils.data.get_data(): dataset dt is stored at twice the
        # simulation sample spacing for these hover datasets.
        dt = 0.5 * float(np.asarray(data["dt"])[index])
    return np.stack(columns, axis=1), actions, dt, expanded


def _norm_vectors(labels: list[str], module_name: str) -> tuple[np.ndarray, np.ndarray]:
    module = importlib.import_module(module_name)
    mins: list[float] = []
    maxs: list[float] = []
    for label in labels:
        if label.startswith("omega"):
            mins.append(float(module.GLOBAL_MIN["omega_min"]))
            maxs.append(float(module.GLOBAL_MAX["omega_max"]))
        else:
            mins.append(float(module.GLOBAL_MIN[label]))
            maxs.append(float(module.GLOBAL_MAX[label]))
    return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


def _rpm(actions: np.ndarray) -> np.ndarray:
    return OMEGA_MIN + np.clip(actions, 0.0, 1.0) * (OMEGA_MAX - OMEGA_MIN)


def _teacher_forced_actions(
    model: torch.nn.Module,
    raw_inputs: np.ndarray,
    labels: list[str],
    normalization_module: str,
    dt: float | None,
) -> np.ndarray:
    mins, maxs = _norm_vectors(labels, normalization_module)
    hx = None
    outputs: list[np.ndarray] = []
    dt_tensor = None if dt is None else torch.tensor(dt, dtype=torch.float32).reshape(1, 1, 1)
    for row in raw_inputs:
        normalized = (row.astype(np.float32) - mins) / (maxs - mins + np.float32(1.0e-10))
        obs = make_model_observation(normalized.reshape(1, -1), use_sequencing=True, device=torch.device("cpu"))
        with torch.no_grad():
            result = model(obs, hx=hx, timespans=dt_tensor)
            prediction, hx = result if isinstance(result, tuple) else (result, None)
        outputs.append(torch.clamp(prediction, 0.0, 1.0).reshape(-1).cpu().numpy())
    return np.asarray(outputs, dtype=np.float64)


def _closed_loop_actions(
    model: torch.nn.Module,
    initial_state: np.ndarray,
    input_labels: list[str],
    normalization_module: str,
    dt: float,
    steps: int,
    integration_method: str = "rk4",
    pass_timespan: bool = True,
) -> np.ndarray:
    labels = expand_feature_labels([label for label in input_labels if label not in {"t", "dt"}])
    mins, maxs = _norm_vectors(labels, normalization_module)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    hx = None
    prev_deriv = None
    actions: list[np.ndarray] = []
    dt_tensor = torch.tensor(dt, dtype=torch.float32).reshape(1, 1, 1) if pass_timespan else None
    for _ in range(steps):
        raw_input = build_input_vector(state, input_labels_without_dt(input_labels))
        normalized = (raw_input.astype(np.float32) - mins) / (maxs - mins + np.float32(1.0e-10))
        obs = make_model_observation(normalized.reshape(1, -1), use_sequencing=True, device=torch.device("cpu"))
        with torch.no_grad():
            result = model(obs, hx=hx, timespans=dt_tensor)
            prediction, hx = result if isinstance(result, tuple) else (result, None)
        action = torch.clamp(prediction, 0.0, 1.0).reshape(-1).cpu().numpy().astype(np.float64)
        actions.append(action)
        state, prev_deriv = integrate_state(integration_method, state, action, dt, prev_deriv=prev_deriv, implicit_iters=1)
    return np.asarray(actions, dtype=np.float64)


def input_labels_without_dt(input_labels: list[str]) -> list[str]:
    return [label for label in input_labels if label not in {"t", "dt"}]


def _plot_motor_rpm(path: Path, time: np.ndarray, series: list[tuple[str, np.ndarray]], title: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for motor_idx, ax in enumerate(axes):
        for label, rpm in series:
            ax.plot(time[: rpm.shape[0]], rpm[: time.shape[0], motor_idx], linewidth=1.2, label=label)
        ax.set_ylabel(f"motor {motor_idx + 1}\n[rpm]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_two_column_motor_rpm(
    path: Path,
    left_title: str,
    right_title: str,
    left_time_series: list[tuple[str, np.ndarray, np.ndarray, str]],
    right_time_series: list[tuple[str, np.ndarray, np.ndarray, str]],
    title: str,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16, 10), sharex="col")
    for col_idx, (col_title, series) in enumerate(
        [(left_title, left_time_series), (right_title, right_time_series)]
    ):
        axes[0, col_idx].set_title(col_title)
        for motor_idx in range(4):
            ax = axes[motor_idx, col_idx]
            for label, time, rpm, style in series:
                ax.plot(
                    time[: rpm.shape[0]],
                    rpm[: time.shape[0], motor_idx],
                    style,
                    linewidth=1.2,
                    label=f"{label}_{motor_idx + 1}",
                )
            ax.set_ylabel(f"motor {motor_idx + 1}\nrpm_cmd [RPM]")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)
        axes[-1, col_idx].set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    torch.set_grad_enabled(False)
    bebop1_cfg = load_yaml(BEBOP1_CONFIG)
    bebop2_cfg = load_yaml(BEBOP2_CONFIG)
    bebop1_model = build_lightning_model(bebop1_cfg, BEBOP1_MODEL, PROJECT, torch.device("cpu"))
    bebop2_model = build_lightning_model(bebop2_cfg, BEBOP2_MODEL, PROJECT, torch.device("cpu"))

    bebop1_inputs, bebop1_ref_u, bebop1_dt, bebop1_labels = _load_raw_trajectory(
        BEBOP1_DATASET, BEBOP1_TRAJ, bebop1_cfg["dataset"]["input_labels"]
    )
    bebop2_inputs, bebop2_ref_u, bebop2_dt, bebop2_labels = _load_raw_trajectory(
        BEBOP2_DATASET, BEBOP2_TRAJ, bebop2_cfg["dataset"]["input_labels"]
    )

    bebop1_time = np.arange(bebop1_inputs.shape[0], dtype=np.float64) * bebop1_dt
    bebop2_time = np.arange(bebop2_inputs.shape[0], dtype=np.float64) * bebop2_dt
    bebop1_tf = _teacher_forced_actions(bebop1_model, bebop1_inputs, bebop1_labels, BEBOP1_NORM, None)
    bebop2_tf = _teacher_forced_actions(bebop2_model, bebop2_inputs, bebop2_labels, BEBOP2_NORM, bebop2_dt)
    _plot_two_column_motor_rpm(
        OUT_DIR / "teacher_forced_motor_rpm_bebop1_vs_bebop2_nova_versaoze.png",
        "CFC / original Bebop1 test - idx 245\nteacher-forced dataset states",
        "NOVA_VERSAOZE BBP2 CFC / corrected Bebop2 test - idx 208\nteacher-forced dataset states",
        [
            ("ref rpm_cmd", bebop1_time, _rpm(bebop1_ref_u), "--"),
            ("LNN open-loop rpm_cmd", bebop1_time, _rpm(bebop1_tf), "-"),
        ],
        [
            ("ref rpm_cmd", bebop2_time, _rpm(bebop2_ref_u), "--"),
            ("LNN open-loop rpm_cmd", bebop2_time, _rpm(bebop2_tf), "-"),
        ],
        "Teacher-forced motor commands: Bebop1 CFC vs NOVA_VERSAOZE Bebop2 CFC",
    )

    bebop1_steps_4s = int(round(HORIZON_SECONDS / bebop1_dt))
    bebop2_steps_4s = int(round(HORIZON_SECONDS / bebop2_dt))
    bebop1_initial_state = state_from_input_features(bebop1_inputs[0], bebop1_labels)
    bebop2_initial_state = state_from_input_features(bebop2_inputs[0], bebop2_labels)
    set_dynamics_model("quadrotor_sim_original")
    bebop1_closed_loop = _closed_loop_actions(
        bebop1_model,
        bebop1_initial_state,
        bebop1_cfg["dataset"]["input_labels"],
        BEBOP1_NORM,
        bebop1_dt,
        bebop1_steps_4s,
        pass_timespan=False,
    )
    set_dynamics_model("quadrotor_sim_matlab")
    bebop2_closed_loop = _closed_loop_actions(
        bebop2_model,
        bebop2_initial_state,
        bebop2_cfg["dataset"]["input_labels"],
        BEBOP2_NORM,
        bebop2_dt,
        bebop2_steps_4s,
        pass_timespan=True,
    )
    _plot_two_column_motor_rpm(
        OUT_DIR / "closed_loop_4s_motor_rpm_nova_versaoze.png",
        "CFC / original Bebop1 test - idx 245",
        "NOVA_VERSAOZE BBP2 CFC / corrected Bebop2 test - idx 208",
        [
            ("ref rpm_cmd", bebop1_time, _rpm(bebop1_ref_u), "--"),
            (
                "sim rpm_cmd",
                np.arange(bebop1_steps_4s, dtype=np.float64) * bebop1_dt,
                _rpm(bebop1_closed_loop),
                "-",
            ),
        ],
        [
            ("ref rpm_cmd", bebop2_time, _rpm(bebop2_ref_u), "--"),
            (
                "sim rpm_cmd",
                np.arange(bebop2_steps_4s, dtype=np.float64) * bebop2_dt,
                _rpm(bebop2_closed_loop),
                "-",
            ),
        ],
        "Motor commands converted to RPM: matched original test vs corrected Bebop2 test (4s)",
    )
    print(OUT_DIR / "teacher_forced_motor_rpm_bebop1_vs_bebop2_nova_versaoze.png")
    print(OUT_DIR / "closed_loop_4s_motor_rpm_nova_versaoze.png")


if __name__ == "__main__":
    main()
