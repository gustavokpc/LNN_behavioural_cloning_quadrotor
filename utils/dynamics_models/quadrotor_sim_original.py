#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Original reduced quadrotor dynamics used by the Python simulators."""

from __future__ import annotations

import numpy as np

from . import DynamicsInfo


G = 9.81
IXX = 0.000906
IYY = 0.001242
IZZ = 0.002054
KX = 1.07933887e-05
KY = 9.65250793e-06
KZ = 2.7862899e-05
KOMEGA = 4.36301076e-08
KH = 0.06255013
KP = 1.4119331e-09
KPV = -0.00797102
KQ = 1.21601884e-09
KQV = 0.01292637
KR1 = 2.57035545e-06
KR2 = 4.10923364e-07
KRR = 0.00081293
OMEGA_MAX = 10000.0
OMEGA_MIN = 5000.0
TAU = 0.06

INFO = DynamicsInfo(
    name="quadrotor_sim_original",
    description="Original reduced quadrotor dynamics equation.",
    omega_min=OMEGA_MIN,
    omega_max=OMEGA_MAX,
    tau=TAU,
    hover_omega=(G / (4.0 * KOMEGA)) ** 0.5,
)


def dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    (
        dx, dy, dz, vx, vy, vz,
        phi, theta, psi, p, q, r,
        mx, my, mz, omega1, omega2, omega3, omega4,
    ) = state
    u1, u2, u3, u4 = action

    d_dx = -q * dz + r * dy - vx
    d_dy = p * dz - r * dx - vy
    d_dz = -p * dy + q * dx - vz

    omegas = omega1 + omega2 + omega3 + omega4
    omegas2 = omega1**2 + omega2**2 + omega3**2 + omega4**2

    d_vx = -q * vz + r * vy - G * np.sin(theta) - KX * omegas * vx
    d_vy = p * vz - r * vx + G * np.cos(theta) * np.sin(phi) - KY * omegas * vy
    d_vz = (
        -p * vy + q * vx + G * np.cos(theta) * np.cos(phi)
        - KZ * omegas * vz - KOMEGA * omegas2 - KH * (vx**2 + vy**2)
    )

    d_phi = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
    d_theta = q * np.cos(phi) - r * np.sin(phi)
    d_psi = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)

    d_omega1 = (OMEGA_MIN + u1 * (OMEGA_MAX - OMEGA_MIN) - omega1) / TAU
    d_omega2 = (OMEGA_MIN + u2 * (OMEGA_MAX - OMEGA_MIN) - omega2) / TAU
    d_omega3 = (OMEGA_MIN + u3 * (OMEGA_MAX - OMEGA_MIN) - omega3) / TAU
    d_omega4 = (OMEGA_MIN + u4 * (OMEGA_MAX - OMEGA_MIN) - omega4) / TAU

    tau_x = KP * (omega1**2 - omega2**2 - omega3**2 + omega4**2) + KPV * vy + mx
    tau_y = KQ * (omega1**2 + omega2**2 - omega3**2 - omega4**2) + KQV * vx + my
    tau_z = (
        KR1 * (-omega1 + omega2 - omega3 + omega4)
        + KR2 * (-d_omega1 + d_omega2 - d_omega3 + d_omega4)
        - KRR * r + mz
    )

    d_p = (q * r * (IYY - IZZ) + tau_x) / IXX
    d_q = (p * r * (IZZ - IXX) + tau_y) / IYY
    d_r = (p * q * (IXX - IYY) + tau_z) / IZZ

    return np.array(
        [
            d_dx, d_dy, d_dz,
            d_vx, d_vy, d_vz,
            d_phi, d_theta, d_psi,
            d_p, d_q, d_r,
            0.0, 0.0, 0.0,
            d_omega1, d_omega2, d_omega3, d_omega4,
        ],
        dtype=np.float64,
    )
