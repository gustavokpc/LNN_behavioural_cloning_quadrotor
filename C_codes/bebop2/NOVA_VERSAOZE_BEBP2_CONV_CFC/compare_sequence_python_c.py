#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

C_DIR = Path(__file__).resolve().parent
PROJECT = C_DIR.parents[2]
REPO_ROOT = PROJECT.parent
sys.path.insert(0, str(REPO_ROOT))

from LNN_behavioural_cloning_quadrotor.utils.config import load_yaml, resolve_checkpoint, resolve_saved_config
from LNN_behavioural_cloning_quadrotor.utils.data import expand_feature_labels, get_norm_vectors
from LNN_behavioural_cloning_quadrotor.utils.quadrotor_sim import build_lightning_model, make_model_observation

MODEL_NAME = "NOVA_VERSAOZE_BEBP2_conv_cfc_default_n64_bebop2_epoch=19_val_loss=0.000098.ckpt"
CONFIG_PATH = resolve_saved_config(MODEL_NAME, PROJECT / "configs")
DATASET_PATH = PROJECT / "datasets" / "hover_dataset_test_bebop2_corrected.npz"
NORMALIZATION_LIMITS = "bebop2_tau_0_06"
DEFAULT_OUTPUT_DIR = PROJECT / "C_codes" / "bebop2" / "NOVA_VERSAOZE_BEBP2_CONV_CFC" / "sequence_compare_outputs"


def _build_shared_library() -> Path:
    lib_path = C_DIR / "libcontroller.so"
    subprocess.run(
        [
            "gcc",
            "-std=c99",
            "-O2",
            "-fPIC",
            "-shared",
            str(C_DIR / "nn_operations.c"),
            str(C_DIR / "nn_parameters.c"),
            "-lm",
            "-o",
            str(lib_path),
        ],
        check=True,
    )
    return lib_path


def _load_c_lib(rebuild: bool) -> ctypes.CDLL:
    lib_path = _build_shared_library() if rebuild or not (C_DIR / "libcontroller.so").exists() else C_DIR / "libcontroller.so"
    lib = ctypes.CDLL(str(lib_path))
    ptr = ctypes.POINTER(ctypes.c_float)
    lib.nn_reset.argtypes = []
    lib.nn_reset.restype = None
    lib.nn_set_timespan.argtypes = [ctypes.c_float]
    lib.nn_set_timespan.restype = None
    lib.nn_control.argtypes = [ptr, ptr]
    lib.nn_control.restype = None
    lib.nn_hidden_size.argtypes = []
    lib.nn_hidden_size.restype = ctypes.c_int
    lib.nn_get_hidden.argtypes = [ptr]
    lib.nn_get_hidden.restype = None
    return lib


def _norm_vectors(labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mins, maxs = get_norm_vectors(labels, NORMALIZATION_LIMITS)
    return mins.reshape(-1).astype(np.float32), maxs.reshape(-1).astype(np.float32)


def _dataset_sequence(dataset_path: Path, config: dict, trajectory: int, steps: int | None) -> tuple[np.ndarray, float, list[str]]:
    labels = [label for label in config["dataset"]["input_labels"] if label not in {"t", "dt"}]
    expanded = expand_feature_labels(labels)
    with np.load(dataset_path) as data:
        max_steps = int(data["dx"].shape[1])
        if trajectory < 0 or trajectory >= int(data["dx"].shape[0]):
            raise IndexError(f"trajectory must be in [0, {data['dx'].shape[0] - 1}], got {trajectory}.")
        n_steps = max_steps if steps is None else min(int(steps), max_steps)
        sequence = np.zeros((n_steps, len(expanded)), dtype=np.float32)
        for col, label in enumerate(expanded):
            if label.startswith("omega"):
                motor_idx = int(label[-1]) - 1
                sequence[:, col] = data["omega"][trajectory, :n_steps, motor_idx]
            elif label in {"Mx_ext", "My_ext", "Mz_ext"}:
                sequence[:, col] = float(data[label][trajectory])
            else:
                sequence[:, col] = data[label][trajectory, :n_steps]
        # get_data() halves this value when creating the dt model channel.
        timespan = float(data["dt"][trajectory]) / 2.0
    return sequence, timespan, expanded


def _run_python(sequence: np.ndarray, timespan: float, config: dict) -> tuple[np.ndarray, np.ndarray]:
    model = build_lightning_model(config, str(resolve_checkpoint(MODEL_NAME, PROJECT)), PROJECT, torch.device("cpu"))
    labels = [label for label in config["dataset"]["input_labels"] if label not in {"t", "dt"}]
    norm_min, norm_max = _norm_vectors(labels)
    normalized = (sequence - norm_min) / (norm_max - norm_min + np.float32(1.0e-10))
    hx = None
    outputs = []
    hidden = []
    dt_tensor = torch.tensor(timespan, dtype=torch.float32).reshape(1, 1, 1)
    with torch.no_grad():
        for row in normalized:
            obs = make_model_observation(row.reshape(1, -1), use_sequencing=True, device=torch.device("cpu"))
            output = model(obs, hx=hx, timespans=dt_tensor)
            prediction, hx = output if isinstance(output, tuple) else (output, None)
            outputs.append(torch.clamp(prediction, 0.0, 1.0).reshape(-1).cpu().numpy())
            hidden.append(hx.reshape(-1).cpu().numpy())
    return np.asarray(outputs, dtype=np.float32), np.asarray(hidden, dtype=np.float32)


def _run_c(sequence: np.ndarray, timespan: float, rebuild: bool) -> tuple[np.ndarray, np.ndarray]:
    lib = _load_c_lib(rebuild)
    hidden_size = int(lib.nn_hidden_size())
    lib.nn_reset()
    lib.nn_set_timespan(ctypes.c_float(timespan))
    outputs = []
    hidden = []
    for row in np.asarray(sequence, dtype=np.float32):
        y = np.zeros(4, dtype=np.float32)
        h = np.zeros(hidden_size, dtype=np.float32)
        lib.nn_control(
            row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        lib.nn_get_hidden(h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        outputs.append(y.copy())
        hidden.append(h.copy())
    return np.asarray(outputs, dtype=np.float32), np.asarray(hidden, dtype=np.float32)


def _write_csv(path: Path, py_output: np.ndarray, c_output: np.ndarray, py_hidden: np.ndarray, c_hidden: np.ndarray) -> None:
    import csv

    header = ["step"]
    header += [f"python_u{i + 1}" for i in range(py_output.shape[1])]
    header += [f"c_u{i + 1}" for i in range(c_output.shape[1])]
    header += [f"abs_err_u{i + 1}" for i in range(c_output.shape[1])]
    header += [f"python_h{i:02d}" for i in range(py_hidden.shape[1])]
    header += [f"c_h{i:02d}" for i in range(c_hidden.shape[1])]
    header += [f"abs_err_h{i:02d}" for i in range(c_hidden.shape[1])]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for step in range(py_output.shape[0]):
            writer.writerow(
                [
                    step,
                    *py_output[step],
                    *c_output[step],
                    *np.abs(py_output[step] - c_output[step]),
                    *py_hidden[step],
                    *c_hidden[step],
                    *np.abs(py_hidden[step] - c_hidden[step]),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare NOVA_VERSAOZE PyTorch and C outputs/hidden states on a dataset sequence.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(CONFIG_PATH)
    sequence, timespan, labels = _dataset_sequence(args.dataset, config, args.trajectory, args.steps)
    py_output, py_hidden = _run_python(sequence, timespan, config)
    c_output, c_hidden = _run_c(sequence, timespan, args.rebuild)

    output_err = np.abs(py_output - c_output)
    hidden_err = np.abs(py_hidden - c_hidden)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / f"nova_versaoze_traj{args.trajectory:03d}_steps{sequence.shape[0]:03d}_sequence_compare.npz"
    csv_path = npz_path.with_suffix(".csv")
    np.savez(
        npz_path,
        input_sequence=sequence,
        input_labels=np.asarray(labels),
        timespan=np.asarray(timespan, dtype=np.float32),
        python_output=py_output,
        c_output=c_output,
        output_abs_error=output_err,
        python_hidden=py_hidden,
        c_hidden=c_hidden,
        hidden_abs_error=hidden_err,
    )
    _write_csv(csv_path, py_output, c_output, py_hidden, c_hidden)

    print(f"Dataset: {args.dataset}")
    print(f"Trajectory: {args.trajectory}")
    print(f"Steps: {sequence.shape[0]}")
    print(f"Timespan passed to CfC: {timespan:.9f}")
    print(f"Final python output: {py_output[-1]}")
    print(f"Final C output: {c_output[-1]}")
    print(f"Final output abs error: {output_err[-1]}")
    print(f"Max output abs error: {float(output_err.max()):.9g}")
    print(f"Final hidden first 8 python: {py_hidden[-1, :8]}")
    print(f"Final hidden first 8 C: {c_hidden[-1, :8]}")
    print(f"Final hidden first 8 abs error: {hidden_err[-1, :8]}")
    print(f"Max hidden abs error: {float(hidden_err.max()):.9g}")
    print(f"Saved NPZ: {npz_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
