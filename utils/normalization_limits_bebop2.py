# -*- coding: utf-8 -*-
"""Bebop2 normalization limits and dynamics constants.

The normalization bounds come from the Bebop2 dataset. Physical parameters are
mirrored from ``utils.dynamics_models.quadrotor_sim_matlab``.
"""

import torch

from .dynamics_models import quadrotor_sim_matlab as bebop2


_BEBOP2_MIN = {
    "dx": -7.07128185,
    "dy": -7.05328296,
    "dz": -5.56019598,
    "vx": -4.62598611,
    "vy": -5.77716991,
    "vz": -3.09426852,
    "phi": -1.14116707,
    "theta": -1.39626251,
    "psi": -3.95168293,
    "p": -9.31929447,
    "q": -10.51657543,
    "r": -3.89352471,
    "Mx_ext": -0.03999910,
    "My_ext": -0.03999795,
    "Mz_ext": -0.00999981,
    "omega_min": 5000.01181729,
}

_BEBOP2_MAX = {
    "dx": 6.99678094,
    "dy": 7.01479211,
    "dz": 5.54890019,
    "vx": 4.70332355,
    "vy": 5.89205006,
    "vz": 3.25420151,
    "phi": 1.36113735,
    "theta": 1.21683973,
    "psi": 3.90837686,
    "p": 9.65025755,
    "q": 10.82619475,
    "r": 4.01194751,
    "Mx_ext": 0.03999963,
    "My_ext": 0.03999934,
    "Mz_ext": 0.00999978,
    "omega_max": 10000.00003220,
}

GLOBAL_MIN = _BEBOP2_MIN
GLOBAL_MAX = _BEBOP2_MAX

G = torch.tensor(bebop2.G, dtype=torch.float32)
MASS = torch.tensor(bebop2.MASS, dtype=torch.float32)
IXX = torch.tensor(bebop2.IXX, dtype=torch.float32)
IYY = torch.tensor(bebop2.IYY, dtype=torch.float32)
IZZ = torch.tensor(bebop2.IZZ, dtype=torch.float32)
OMEGA_MAX = torch.tensor(bebop2.OMEGA_MAX, dtype=torch.float32)
OMEGA_MIN = torch.tensor(bebop2.OMEGA_MIN, dtype=torch.float32)
TAU = torch.tensor(bebop2.TAU, dtype=torch.float32)

RHO = torch.tensor(bebop2.RHO, dtype=torch.float32)
R = torch.tensor(bebop2.R, dtype=torch.float32)
L = torch.tensor(bebop2.L, dtype=torch.float32)
B = torch.tensor(bebop2.B, dtype=torch.float32)
AREA = torch.tensor(bebop2.AREA, dtype=torch.float32)
S = torch.tensor(bebop2.S, dtype=torch.float32)
RPM_TO_RAD_S = torch.tensor(bebop2.RPM_TO_RAD_S, dtype=torch.float32)
