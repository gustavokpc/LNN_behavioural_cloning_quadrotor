#!/usr/bin/env python3
"""Compile the export and compare a recurrent input sequence with PyTorch."""

from __future__ import annotations

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

from LNN_behavioural_cloning_quadrotor.utils.config import load_yaml
from LNN_behavioural_cloning_quadrotor.utils.data import get_norm_vectors
from LNN_behavioural_cloning_quadrotor.utils.quadrotor_sim import (
    build_lightning_model,
    make_model_observation,
)


CHECKPOINT = PROJECT / "checkpoints" / "bebop2" / "conv_cfc_default_n64_bebop2_no_dt_delayed.ckpt"
CONFIG = PROJECT / "configs" / "bebop2" / "conv_cfc_default_n64_bebop2_no_dt_delayed.yaml"
LIBRARY = Path("/tmp/libcontroller_bebop2_no_dt_delayed.so")
PROFILE = "bebop2_tau_0_06"


def build_c() -> None:
    subprocess.run(
        [
            "gcc",
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fPIC",
            "-shared",
            str(C_DIR / "nn_cfc_operations.c"),
            str(C_DIR / "nn_cfc_parameters.c"),
            "-lm",
            "-o",
            str(LIBRARY),
        ],
        check=True,
    )


def test_states(count: int = 100) -> np.ndarray:
    rng = np.random.default_rng(12345)
    low, high = get_norm_vectors(
        ["dx", "dy", "dz", "vx", "vy", "vz", "phi", "theta", "psi",
         "p", "q", "r", "Mx_ext", "My_ext", "Mz_ext", "omega"],
        PROFILE,
    )
    low = low.reshape(-1).astype(np.float32)
    high = high.reshape(-1).astype(np.float32)
    normalized = rng.uniform(0.1, 0.9, size=(count, low.size)).astype(np.float32)
    return low + normalized * (high - low)


def python_outputs(states: np.ndarray) -> np.ndarray:
    config = load_yaml(CONFIG)
    model = build_lightning_model(config, str(CHECKPOINT), PROJECT, torch.device("cpu"))
    low, high = get_norm_vectors(
        ["dx", "dy", "dz", "vx", "vy", "vz", "phi", "theta", "psi",
         "p", "q", "r", "Mx_ext", "My_ext", "Mz_ext", "omega"],
        PROFILE,
    )
    low = low.reshape(-1).astype(np.float32)
    high = high.reshape(-1).astype(np.float32)
    hx = None
    outputs = []
    for state in states:
        normalized = (state - low) / (high - low + 1.0e-10)
        obs = make_model_observation(
            normalized.reshape(1, -1),
            use_sequencing=True,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            prediction, hx = model(obs, hx=hx, timespans=None)
        outputs.append(torch.clamp(prediction, 0.0, 1.0).reshape(-1).numpy())
    return np.asarray(outputs, dtype=np.float32)


def c_outputs(states: np.ndarray) -> np.ndarray:
    library = ctypes.CDLL(str(LIBRARY))
    float_pointer = ctypes.POINTER(ctypes.c_float)
    library.nn_reset.argtypes = []
    library.nn_reset.restype = None
    library.nn_control.argtypes = [float_pointer, float_pointer]
    library.nn_control.restype = None
    library.nn_reset()
    outputs = []
    for state in states:
        state = np.ascontiguousarray(state, dtype=np.float32)
        output = np.empty(4, dtype=np.float32)
        library.nn_control(
            state.ctypes.data_as(float_pointer),
            output.ctypes.data_as(float_pointer),
        )
        outputs.append(output.copy())
    return np.asarray(outputs, dtype=np.float32)


def main() -> None:
    build_c()
    states = test_states()
    expected = python_outputs(states)
    actual = c_outputs(states)
    error = np.abs(expected - actual)
    print(f"compared outputs: {error.size}")
    print(f"max abs error: {error.max():.9g}")
    print(f"mean abs error: {error.mean():.9g}")
    if not np.allclose(expected, actual, atol=2.0e-5, rtol=2.0e-5):
        raise SystemExit("Python/C comparison failed")
    print("Python/C comparison passed")


if __name__ == "__main__":
    main()
