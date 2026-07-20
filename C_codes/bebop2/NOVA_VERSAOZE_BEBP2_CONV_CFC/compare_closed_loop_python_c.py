#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from LNN_behavioural_cloning_quadrotor.utils.dynamics_models import get_dynamics_model, set_dynamics_model
from LNN_behavioural_cloning_quadrotor.utils.quadrotor_sim import (
    STATE_INDEX,
    build_input_vector,
    build_lightning_model,
    integrate_state,
    make_model_observation,
)

MODEL_NAME = "NOVA_VERSAOZE_BEBP2_conv_cfc_default_n64_bebop2_epoch=19_val_loss=0.000098.ckpt"
CONFIG_PATH = resolve_saved_config(MODEL_NAME, PROJECT / "configs")
DATASET_PATH = PROJECT / "datasets" / "hover_dataset_test_bebop2_corrected.npz"
NORMALIZATION_LIMITS = "bebop2_tau_0_06"
DEFAULT_OUTPUT_DIR = C_DIR / "closed_loop_compare_outputs"


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


def _override_tau(tau: float | None) -> None:
    if tau is None:
        return
    model = get_dynamics_model()
    if not hasattr(model, "TAU"):
        raise AttributeError(f"Dynamics model {model.__name__} does not expose TAU.")
    model.TAU = float(tau)


def _norm_vectors(labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mins, maxs = get_norm_vectors(labels, NORMALIZATION_LIMITS)
    return mins.reshape(-1).astype(np.float32), maxs.reshape(-1).astype(np.float32)


def _initial_state_from_dataset(dataset_path: Path, trajectory: int) -> tuple[np.ndarray, float]:
    with np.load(dataset_path) as data:
        if trajectory < 0 or trajectory >= int(data["dx"].shape[0]):
            raise IndexError(f"trajectory must be in [0, {data['dx'].shape[0] - 1}], got {trajectory}.")
        state = np.zeros(19, dtype=np.float64)
        scalar_labels = [
            "dx",
            "dy",
            "dz",
            "vx",
            "vy",
            "vz",
            "phi",
            "theta",
            "psi",
            "p",
            "q",
            "r",
        ]
        for label in scalar_labels:
            state[STATE_INDEX[label]] = float(data[label][trajectory, 0])
        for label in ["Mx_ext", "My_ext", "Mz_ext"]:
            state[STATE_INDEX[label]] = float(data[label][trajectory])
        state[STATE_INDEX["omega1"] : STATE_INDEX["omega4"] + 1] = data["omega"][trajectory, 0, :]
        # Match the training data path, where get_data() halves the stored dt.
        dt = float(data["dt"][trajectory]) / 2.0
    return state, dt


def _run_python(
    initial_state: np.ndarray,
    dt: float,
    max_steps: int,
    dist_error: float,
    config: dict,
    integration_method: str,
    implicit_iters: int,
) -> dict[str, np.ndarray]:
    model = build_lightning_model(config, str(resolve_checkpoint(MODEL_NAME, PROJECT)), PROJECT, torch.device("cpu"))
    labels = [label for label in config["dataset"]["input_labels"] if label not in {"t", "dt"}]
    norm_min, norm_max = _norm_vectors(labels)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    hx = None
    prev_deriv = None
    inputs = []
    outputs = []
    hidden = []
    states = [state.copy()]
    dt_tensor = torch.tensor(dt, dtype=torch.float32).reshape(1, 1, 1)

    for _ in range(max_steps):
        raw_input = build_input_vector(state, labels).astype(np.float32)
        normalized = (raw_input - norm_min) / (norm_max - norm_min + np.float32(1.0e-10))
        obs = make_model_observation(normalized.reshape(1, -1), use_sequencing=True, device=torch.device("cpu"))
        with torch.no_grad():
            output = model(obs, hx=hx, timespans=dt_tensor)
            prediction, hx = output if isinstance(output, tuple) else (output, None)
        action = torch.clamp(prediction, 0.0, 1.0).reshape(-1).cpu().numpy().astype(np.float64)
        inputs.append(raw_input)
        outputs.append(action.astype(np.float32))
        hidden.append(hx.reshape(-1).cpu().numpy().astype(np.float32))
        state, prev_deriv = integrate_state(
            integration_method,
            state,
            action,
            dt,
            prev_deriv=prev_deriv,
            implicit_iters=implicit_iters,
        )
        states.append(state.copy())
        if np.linalg.norm(state[0:3]) < dist_error:
            break

    return {
        "inputs": np.asarray(inputs, dtype=np.float32),
        "outputs": np.asarray(outputs, dtype=np.float32),
        "hidden": np.asarray(hidden, dtype=np.float32),
        "states": np.asarray(states, dtype=np.float64),
    }


def _run_c(
    initial_state: np.ndarray,
    dt: float,
    max_steps: int,
    dist_error: float,
    config: dict,
    integration_method: str,
    implicit_iters: int,
    rebuild: bool,
) -> dict[str, np.ndarray]:
    lib = _load_c_lib(rebuild)
    hidden_size = int(lib.nn_hidden_size())
    labels = [label for label in config["dataset"]["input_labels"] if label not in {"t", "dt"}]
    state = np.asarray(initial_state, dtype=np.float64).copy()
    prev_deriv = None
    inputs = []
    outputs = []
    hidden = []
    states = [state.copy()]
    lib.nn_reset()
    lib.nn_set_timespan(ctypes.c_float(dt))

    for _ in range(max_steps):
        raw_input = build_input_vector(state, labels).astype(np.float32)
        action = np.zeros(4, dtype=np.float32)
        h = np.zeros(hidden_size, dtype=np.float32)
        lib.nn_control(
            raw_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            action.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        lib.nn_get_hidden(h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        inputs.append(raw_input)
        outputs.append(action.copy())
        hidden.append(h.copy())
        state, prev_deriv = integrate_state(
            integration_method,
            state,
            action.astype(np.float64),
            dt,
            prev_deriv=prev_deriv,
            implicit_iters=implicit_iters,
        )
        states.append(state.copy())
        if np.linalg.norm(state[0:3]) < dist_error:
            break

    return {
        "inputs": np.asarray(inputs, dtype=np.float32),
        "outputs": np.asarray(outputs, dtype=np.float32),
        "hidden": np.asarray(hidden, dtype=np.float32),
        "states": np.asarray(states, dtype=np.float64),
    }


def _run_shared_state(
    initial_state: np.ndarray,
    dt: float,
    max_steps: int,
    dist_error: float,
    config: dict,
    integration_method: str,
    implicit_iters: int,
    rebuild: bool,
    state_source: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    py_model = build_lightning_model(config, str(resolve_checkpoint(MODEL_NAME, PROJECT)), PROJECT, torch.device("cpu"))
    c_lib = _load_c_lib(rebuild)
    hidden_size = int(c_lib.nn_hidden_size())
    labels = [label for label in config["dataset"]["input_labels"] if label not in {"t", "dt"}]
    norm_min, norm_max = _norm_vectors(labels)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    py_hx = None
    prev_deriv = None
    py_inputs = []
    c_inputs = []
    py_outputs = []
    c_outputs = []
    py_hidden = []
    c_hidden = []
    states = [state.copy()]
    dt_tensor = torch.tensor(dt, dtype=torch.float32).reshape(1, 1, 1)
    c_lib.nn_reset()
    c_lib.nn_set_timespan(ctypes.c_float(dt))

    for _ in range(max_steps):
        raw_input = build_input_vector(state, labels).astype(np.float32)
        normalized = (raw_input - norm_min) / (norm_max - norm_min + np.float32(1.0e-10))
        obs = make_model_observation(normalized.reshape(1, -1), use_sequencing=True, device=torch.device("cpu"))
        with torch.no_grad():
            py_output = py_model(obs, hx=py_hx, timespans=dt_tensor)
            py_prediction, py_hx = py_output if isinstance(py_output, tuple) else (py_output, None)
        py_action = torch.clamp(py_prediction, 0.0, 1.0).reshape(-1).cpu().numpy().astype(np.float64)

        c_action = np.zeros(4, dtype=np.float32)
        c_h = np.zeros(hidden_size, dtype=np.float32)
        c_lib.nn_control(
            raw_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c_action.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        c_lib.nn_get_hidden(c_h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        py_inputs.append(raw_input)
        c_inputs.append(raw_input)
        py_outputs.append(py_action.astype(np.float32))
        c_outputs.append(c_action.copy())
        py_hidden.append(py_hx.reshape(-1).cpu().numpy().astype(np.float32))
        c_hidden.append(c_h.copy())

        action_for_dynamics = py_action if state_source == "python" else c_action.astype(np.float64)
        state, prev_deriv = integrate_state(
            integration_method,
            state,
            action_for_dynamics,
            dt,
            prev_deriv=prev_deriv,
            implicit_iters=implicit_iters,
        )
        states.append(state.copy())
        if np.linalg.norm(state[0:3]) < dist_error:
            break

    py = {
        "inputs": np.asarray(py_inputs, dtype=np.float32),
        "outputs": np.asarray(py_outputs, dtype=np.float32),
        "hidden": np.asarray(py_hidden, dtype=np.float32),
        "states": np.asarray(states, dtype=np.float64),
    }
    c = {
        "inputs": np.asarray(c_inputs, dtype=np.float32),
        "outputs": np.asarray(c_outputs, dtype=np.float32),
        "hidden": np.asarray(c_hidden, dtype=np.float32),
        "states": np.asarray(states, dtype=np.float64),
    }
    return py, c


def _write_csv(path: Path, py: dict[str, np.ndarray], c: dict[str, np.ndarray]) -> None:
    n_steps = min(len(py["outputs"]), len(c["outputs"]))
    header = ["step", "py_dist", "c_dist"]
    header += [f"python_u{i + 1}" for i in range(4)]
    header += [f"c_u{i + 1}" for i in range(4)]
    header += [f"abs_err_u{i + 1}" for i in range(4)]
    header += [f"python_h{i:02d}" for i in range(py["hidden"].shape[1])]
    header += [f"c_h{i:02d}" for i in range(c["hidden"].shape[1])]
    header += [f"abs_err_h{i:02d}" for i in range(c["hidden"].shape[1])]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for step in range(n_steps):
            writer.writerow(
                [
                    step,
                    float(np.linalg.norm(py["states"][step, 0:3])),
                    float(np.linalg.norm(c["states"][step, 0:3])),
                    *py["outputs"][step],
                    *c["outputs"][step],
                    *np.abs(py["outputs"][step] - c["outputs"][step]),
                    *py["hidden"][step],
                    *c["hidden"][step],
                    *np.abs(py["hidden"][step] - c["hidden"][step]),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop NOVA_VERSAOZE PyTorch vs C comparison including hidden state.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--dist-error", type=float, default=0.1)
    parser.add_argument("--dynamics-model", default="quadrotor_sim_matlab")
    parser.add_argument("--tau", type=float, default=0.06)
    parser.add_argument("--integration-method", default="rk4")
    parser.add_argument("--implicit-iters", type=int, default=1)
    parser.add_argument(
        "--comparison-mode",
        choices=["independent", "shared-python-state", "shared-c-state"],
        default="independent",
        help=(
            "independent runs separate closed loops; shared-* feeds both controllers the same "
            "generated state and uses that controller's action to advance the shared dynamics."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(CONFIG_PATH)
    set_dynamics_model(args.dynamics_model)
    _override_tau(args.tau)
    initial_state, dt = _initial_state_from_dataset(args.dataset, args.trajectory)
    if args.comparison_mode == "independent":
        py = _run_python(
            initial_state,
            dt,
            args.max_steps,
            args.dist_error,
            config,
            args.integration_method,
            args.implicit_iters,
        )
        c = _run_c(
            initial_state,
            dt,
            args.max_steps,
            args.dist_error,
            config,
            args.integration_method,
            args.implicit_iters,
            args.rebuild,
        )
    else:
        state_source = "python" if args.comparison_mode == "shared-python-state" else "c"
        py, c = _run_shared_state(
            initial_state,
            dt,
            args.max_steps,
            args.dist_error,
            config,
            args.integration_method,
            args.implicit_iters,
            args.rebuild,
            state_source,
        )

    n_steps = min(len(py["outputs"]), len(c["outputs"]))
    output_err = np.abs(py["outputs"][:n_steps] - c["outputs"][:n_steps])
    hidden_err = np.abs(py["hidden"][:n_steps] - c["hidden"][:n_steps])
    state_err = np.abs(py["states"][: n_steps + 1] - c["states"][: n_steps + 1])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"nova_versaoze_traj{args.trajectory:03d}_{args.comparison_mode}_closed_loop_{n_steps:03d}steps"
    npz_path = args.output_dir / f"{stem}.npz"
    csv_path = args.output_dir / f"{stem}.csv"
    np.savez(
        npz_path,
        initial_state=initial_state,
        dt=np.asarray(dt, dtype=np.float32),
        python_inputs=py["inputs"],
        c_inputs=c["inputs"],
        python_outputs=py["outputs"],
        c_outputs=c["outputs"],
        python_hidden=py["hidden"],
        c_hidden=c["hidden"],
        python_states=py["states"],
        c_states=c["states"],
        output_abs_error=output_err,
        hidden_abs_error=hidden_err,
        state_abs_error=state_err,
    )
    _write_csv(csv_path, py, c)

    print(f"Dataset initial state: {args.dataset}")
    print(f"Trajectory: {args.trajectory}")
    print(f"Target: origin in relative/body-frame state, stop dist_error={args.dist_error}")
    print(f"dt/timespan: {dt:.9f}")
    print(f"Dynamics: {args.dynamics_model}, tau={args.tau}, integration={args.integration_method}")
    print(f"Comparison mode: {args.comparison_mode}")
    print(f"Python steps: {len(py['outputs'])}, C steps: {len(c['outputs'])}, compared steps: {n_steps}")
    print(f"Initial distance: {float(np.linalg.norm(initial_state[0:3])):.9f}")
    print(f"Final python distance: {float(np.linalg.norm(py['states'][-1, 0:3])):.9f}")
    print(f"Final C distance: {float(np.linalg.norm(c['states'][-1, 0:3])):.9f}")
    print(f"Final python output: {py['outputs'][n_steps - 1]}")
    print(f"Final C output: {c['outputs'][n_steps - 1]}")
    print(f"Final output abs error: {output_err[-1]}")
    print(f"Max output abs error: {float(output_err.max()):.9g}")
    print(f"Final hidden first 8 python: {py['hidden'][n_steps - 1, :8]}")
    print(f"Final hidden first 8 C: {c['hidden'][n_steps - 1, :8]}")
    print(f"Final hidden first 8 abs error: {hidden_err[-1, :8]}")
    print(f"Max hidden abs error: {float(hidden_err.max()):.9g}")
    print(f"Max state abs error: {float(state_err.max()):.9g}")
    print(f"Saved NPZ: {npz_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
