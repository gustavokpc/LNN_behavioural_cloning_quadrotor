#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selectable quadrotor dynamics equations for simulation rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType

import numpy as np


@dataclass(frozen=True)
class DynamicsInfo:
    name: str
    description: str
    omega_min: float
    omega_max: float
    tau: float
    hover_omega: float | None = None

    @property
    def omega_mid(self) -> float:
        return 0.5 * (self.omega_max + self.omega_min)

    @property
    def u_hover(self) -> float | None:
        if self.hover_omega is None:
            return None
        return (self.hover_omega - self.omega_min) / (self.omega_max - self.omega_min)


_PACKAGE = __name__
_ACTIVE_MODEL: ModuleType | None = None


def available_dynamics_models() -> list[str]:
    root = Path(__file__).resolve().parent
    names = [
        path.stem
        for path in root.glob("*.py")
        if not path.name.startswith("_") and path.stem != "__init__"
    ]
    return sorted(names)


def load_dynamics_model(name: str) -> ModuleType:
    model_name = name.strip()
    if not model_name:
        raise ValueError("Dynamics model name cannot be empty.")
    try:
        module = import_module(f"{_PACKAGE}.{model_name}")
    except ModuleNotFoundError as exc:
        valid = ", ".join(available_dynamics_models())
        raise KeyError(f"Unknown dynamics model '{name}'. Choose one of: {valid}") from exc
    if not hasattr(module, "INFO") or not hasattr(module, "dynamics"):
        raise AttributeError(f"Dynamics model '{name}' must define INFO and dynamics(state, action).")
    return module


def set_dynamics_model(name: str) -> ModuleType:
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = load_dynamics_model(name)
    return _ACTIVE_MODEL


def get_dynamics_model() -> ModuleType:
    global _ACTIVE_MODEL
    if _ACTIVE_MODEL is None:
        _ACTIVE_MODEL = load_dynamics_model("quadrotor_sim")
    return _ACTIVE_MODEL


def get_dynamics_info() -> DynamicsInfo:
    return get_dynamics_model().INFO


def dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    return get_dynamics_model().dynamics(state, action)
