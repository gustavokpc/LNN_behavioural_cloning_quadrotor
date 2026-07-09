#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ctypes wrapper for the exported C controllers."""

from __future__ import annotations

import ctypes
import re
import subprocess
from pathlib import Path

import numpy as np


CHECKPOINT_TO_C_DIR = {
    "mlp_epoch=19_val_loss=0.003130": "MLP",
    "LTC_64_neurons_seq_1_epoch=18_val_loss=0.000193": "LTC",
    "RNN_64_neurons_seq_1_epoch=17_val_loss=0.000147": "RNN",
    "conv_cfc_default_n64_epoch=17_val_loss=0.000326": "CONV_CFC_DEFAULT",
    "new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142": "CFC",
    "new_CFC_pure_64_neurons_seq_1_epoch=17_val_loss=0.000203": "CFC_PURE",
    "new_CTRNN_64_neurons_seq_1_epoch=19_val_loss=0.000150": "CTRNN",
    "new_GRU_64_neurons_seq_1_epoch=19_val_loss=0.000088": "GRU",
    "new_LSTM_64_neurons_seq_1_epoch=17_val_loss=0.000092": "LSTM",
    "new_NCP_CFC_60_neurons_seq_1_epoch=18_val_loss=0.000143": "NCP_CFC",
    "NOVA_VERSAOZE_BEBP2_conv_cfc_default_n64_bebop2_epoch=19_val_loss=0.000098": "NOVA_VERSAOZE_BEBP2_CONV_CFC",
    "ERRADO_new_conv_bebop2_CFC_64_neurons_epoch=18_val_loss=0.000127": "ERRADO_BEBOP2_CONV_CFC",
}


def c_dir_from_model_path(model_path: str, project_root: Path) -> Path:
    stem = Path(model_path).stem
    folder_name = CHECKPOINT_TO_C_DIR.get(stem)
    if folder_name is None:
        raise KeyError(f"No C export folder is mapped for checkpoint '{stem}'.")
    folder = project_root / "C_codes" / folder_name
    if not folder.is_dir():
        raise FileNotFoundError(f"C export folder not found: {folder}")
    return folder


def _read_define(header: Path, name: str) -> int:
    pattern = re.compile(rf"#define\s+{re.escape(name)}\s+(\d+)")
    for line in header.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            return int(match.group(1))
    raise KeyError(f"Could not find '#define {name}' in {header}.")


class CController:
    """Load and call one exported C controller."""

    def __init__(self, c_dir: Path, rebuild: bool = False) -> None:
        self.c_dir = Path(c_dir)
        self.header_path = self.c_dir / "nn_parameters.h"
        self.num_states = _read_define(self.header_path, "NUM_STATES")
        self.num_controls = _read_define(self.header_path, "NUM_CONTROLS")
        self.lib_path = self.c_dir / "libcontroller.so"
        if rebuild or not self.lib_path.exists():
            self._build_shared_library()
        self.lib = ctypes.CDLL(str(self.lib_path))
        self.lib.nn_reset.argtypes = []
        self.lib.nn_reset.restype = None
        array_in = ctypes.POINTER(ctypes.c_float)
        array_out = ctypes.POINTER(ctypes.c_float)
        self.lib.nn_control.argtypes = [array_in, array_out]
        self.lib.nn_control.restype = None
        self._set_timespan = getattr(self.lib, "nn_set_timespan", None)
        if self._set_timespan is not None:
            self._set_timespan.argtypes = [ctypes.c_float]
            self._set_timespan.restype = None

    def _build_shared_library(self) -> None:
        cmd = [
            "gcc",
            "-std=c99",
            "-O2",
            "-fPIC",
            "-shared",
            str(self.c_dir / "nn_operations.c"),
            str(self.c_dir / "nn_parameters.c"),
            "-lm",
            "-o",
            str(self.lib_path),
        ]
        subprocess.run(cmd, check=True)

    def reset(self) -> None:
        self.lib.nn_reset()

    def set_timespan(self, timespan: float) -> None:
        if self._set_timespan is not None:
            self._set_timespan(ctypes.c_float(float(timespan)))

    def predict(self, controller_input: np.ndarray, timespan: float | None = None) -> np.ndarray:
        x = np.asarray(controller_input, dtype=np.float32).reshape(-1)
        if x.size != self.num_states:
            raise ValueError(f"C controller expects {self.num_states} inputs, got {x.size}.")
        if timespan is not None:
            self.set_timespan(timespan)
        y = np.zeros(self.num_controls, dtype=np.float32)
        self.lib.nn_control(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return y.astype(np.float64)
