# -*- coding: utf-8 -*-
"""Updated Bebop2 normalization limits and dynamics constants.

The normalization bounds come from the updated Bebop2 dataset. Physical
parameters are mirrored from ``utils.dynamics_models.quadrotor_sim_matlab``.
"""

import torch

from .dynamics_models import quadrotor_sim_matlab as bebop2


_BEBOP2_MIN = {
    "dx": -7.07153780,
    "dy": -7.05284256,
    "dz": -5.56030491,
    "vx": -4.63237996,
    "vy": -4.66044117,
    "vz": -1.65918198,
    "phi": -1.14164557,
    "theta": -0.99209529,
    "psi": -3.95187639,
    "p": -9.32495591,
    "q": -9.22844874,
    "r": -3.88378675,
    "Mx_ext": -0.03999910,
    "My_ext": -0.03999795,
    "Mz_ext": -0.00999981,
    "omega_min": 5000.01181729,
}

_BEBOP2_MAX = {
    "dx": 6.99718001,
    "dy": 7.01489217,
    "dz": 5.54857129,
    "vx": 4.61399152,
    "vy": 4.73603565,
    "vz": 1.83613092,
    "phi": 1.05933083,
    "theta": 1.14382555,
    "psi": 3.90809746,
    "p": 9.40760500,
    "q": 8.90656303,
    "r": 4.01474406,
    "Mx_ext": 0.03999963,
    "My_ext": 0.03999934,
    "Mz_ext": 0.00999978,
    "omega_max": 9999.99372531,
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
