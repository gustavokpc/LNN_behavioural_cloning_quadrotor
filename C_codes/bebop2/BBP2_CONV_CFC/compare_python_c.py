#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

C_DIR = Path(__file__).resolve().parent
PROJECT = C_DIR.parents[2]
sys.path.insert(0, str(PROJECT))

from utils.config import load_yaml, resolve_checkpoint, resolve_saved_config
from utils.data import expand_feature_labels, get_norm_vectors
from utils.quadrotor_sim import build_lightning_model, make_model_observation

MODEL_NAME = "bbp2_conv_cfc_default_n64_epoch=19_val_loss=0.000143.ckpt"
STATE = np.array(
    [
        0.3,
        -0.2,
        0.1,
        0.05,
        -0.03,
        0.02,
        0.01,
        -0.02,
        0.03,
        0.1,
        -0.1,
        0.05,
        0.0,
        0.0,
        0.0,
        5500.0,
        5600.0,
        5700.0,
        5800.0,
    ],
    dtype=np.float32,
)
DT = 0.01


def build_c() -> None:
    subprocess.run(
        [
            "gcc",
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-pedantic",
            str(C_DIR / "run_controller.c"),
            str(C_DIR / "nn_operations.c"),
            str(C_DIR / "nn_parameters.c"),
            "-lm",
            "-o",
            str(C_DIR / "run_controller"),
        ],
        check=True,
    )


def py_out() -> np.ndarray:
    cfg = load_yaml(resolve_saved_config(MODEL_NAME, PROJECT / "configs"))
    model = build_lightning_model(cfg, MODEL_NAME, PROJECT, torch.device("cpu"))
    base_labels = expand_feature_labels([label for label in cfg["dataset"]["input_labels"] if label not in {"t", "dt"}])
    norm_min, norm_max = get_norm_vectors(base_labels, "bebop2_tau_0_06")
    norm_min = norm_min.reshape(-1).astype(np.float32)
    norm_max = norm_max.reshape(-1).astype(np.float32)
    x = (STATE - norm_min) / (norm_max - norm_min + 1.0e-10)
    obs = make_model_observation(x.reshape(1, -1), use_sequencing=True, device=torch.device("cpu"))
    dt = torch.tensor(DT, dtype=torch.float32).reshape(1, 1, 1)
    with torch.no_grad():
        y = model(obs, timespans=dt)[0].reshape(-1).cpu().numpy()
    return np.clip(y, 0.0, 1.0)


def c_out() -> np.ndarray:
    res = subprocess.run(
        [str(C_DIR / "run_controller")] + [f"{float(v):.9g}" for v in STATE],
        check=True,
        capture_output=True,
        text=True,
    )
    return np.fromstring(res.stdout, sep=" ", dtype=np.float32)


if __name__ == "__main__":
    build_c()
    yp = py_out()
    yc = c_out()
    err = np.abs(yp - yc)
    print("python:", yp)
    print("c:", yc)
    print("abs error:", err)
    print("max abs error:", float(err.max()))
