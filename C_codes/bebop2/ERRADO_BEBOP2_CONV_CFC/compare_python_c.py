#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

C_DIR = Path(__file__).resolve().parent
PROJECT = C_DIR.parents[2]
REPO_ROOT = PROJECT.parent
sys.path.insert(0, str(REPO_ROOT))

from LNN_behavioural_cloning_quadrotor.utils.c_controller import CController
from LNN_behavioural_cloning_quadrotor.utils.config import load_yaml, resolve_saved_config
from LNN_behavioural_cloning_quadrotor.utils.data import expand_feature_labels, get_norm_vectors
from LNN_behavioural_cloning_quadrotor.utils.quadrotor_sim import build_lightning_model, make_model_observation

MODEL_NAME = "ERRADO_new_conv_bebop2_CFC_64_neurons_epoch=18_val_loss=0.000127.ckpt"
CONFIG_PATH = resolve_saved_config("ERRADO_new_conv_bebop2_CFC_64_neurons_epoch=18_val_loss=0.000127_cfc_compat.yaml", PROJECT / "configs")
NORMALIZATION_LIMITS = "bebop2_tau_0_06"
DT = 0.01
STATE = np.array(
    [0.3, -0.2, 0.1, 0.05, -0.03, 0.02, 0.01, -0.02, 0.03, 0.1, -0.1, 0.05, 0.0, 0.0, 0.0, 5500.0, 5600.0, 5700.0, 5800.0],
    dtype=np.float32,
)


def py_out() -> np.ndarray:
    cfg = load_yaml(CONFIG_PATH)
    model = build_lightning_model(cfg, MODEL_NAME, PROJECT, torch.device("cpu"))
    labels = expand_feature_labels([label for label in cfg["dataset"]["input_labels"] if label not in {"t", "dt"}])
    norm_min, norm_max = get_norm_vectors(labels, NORMALIZATION_LIMITS)
    norm_min = norm_min.reshape(-1).astype(np.float32)
    norm_max = norm_max.reshape(-1).astype(np.float32)
    x = (STATE - norm_min) / (norm_max - norm_min + 1.0e-10)
    obs = make_model_observation(x.reshape(1, -1), use_sequencing=True, device=torch.device("cpu"))
    dt = torch.tensor(DT, dtype=torch.float32).reshape(1, 1, 1)
    with torch.no_grad():
        y = model(obs, timespans=dt)[0].reshape(-1).cpu().numpy()
    return np.clip(y, 0.0, 1.0)


def c_out() -> np.ndarray:
    controller = CController(C_DIR)
    controller.reset()
    return controller.predict(STATE, timespan=DT).astype(np.float32)


if __name__ == "__main__":
    yp = py_out()
    yc = c_out()
    err = np.abs(yp - yc)
    print("python:", yp)
    print("c:", yc)
    print("abs error:", err)
    print("max abs error:", float(err.max()))
