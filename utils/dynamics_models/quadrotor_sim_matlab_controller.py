#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bebop2 Matlab dynamics with yaw convention adapted to the trained controller."""

from __future__ import annotations

import numpy as np

from .quadrotor_sim_matlab import (
    G,
    IXX,
    IYY,
    IZZ,
    MASS,
    OMEGA_MAX,
    OMEGA_MIN,
    TAU,
    INFO as MATLAB_INFO,
    forces_moments,
)
from . import DynamicsInfo


INFO = DynamicsInfo(
    name="quadrotor_sim_matlab_controller",
    description="Bebop2 Matlab force/moment model with rotor yaw sign adapted to the controller-training convention.",
    omega_min=OMEGA_MIN,
    omega_max=OMEGA_MAX,
    tau=TAU,
    hover_omega=MATLAB_INFO.hover_omega,
)


def dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    (
        dx, dy, dz, vx, vy, vz,
        phi, theta, psi, p, q, r,
        mx_ext, my_ext, mz_ext, omega1, omega2, omega3, omega4,
    ) = state
    u1, u2, u3, u4 = action

    d_dx = -q * dz + r * dy - vx
    d_dy = p * dz - r * dx - vy
    d_dz = -p * dy + q * dx - vz

    d_omega1 = (OMEGA_MIN + u1 * (OMEGA_MAX - OMEGA_MIN) - omega1) / TAU
    d_omega2 = (OMEGA_MIN + u2 * (OMEGA_MAX - OMEGA_MIN) - omega2) / TAU
    d_omega3 = (OMEGA_MIN + u3 * (OMEGA_MAX - OMEGA_MIN) - omega3) / TAU
    d_omega4 = (OMEGA_MIN + u4 * (OMEGA_MAX - OMEGA_MIN) - omega4) / TAU

    force_body, moment_body = forces_moments(
        vel=np.asarray([vx, vy, vz], dtype=np.float64),
        rates=np.asarray([p, q, r], dtype=np.float64),
        omega_rpm=np.asarray([omega1, omega2, omega3, omega4], dtype=np.float64),
        rotor_yaw_sign=-1.0,
    )
    moment_body = moment_body + np.asarray([mx_ext, my_ext, mz_ext], dtype=np.float64)

    d_vx = -q * vz + r * vy - G * np.sin(theta) + force_body[0] / MASS
    d_vy = p * vz - r * vx + G * np.cos(theta) * np.sin(phi) + force_body[1] / MASS
    d_vz = -p * vy + q * vx + G * np.cos(theta) * np.cos(phi) + force_body[2] / MASS

    d_phi = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
    d_theta = q * np.cos(phi) - r * np.sin(phi)
    d_psi = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)

    inertia_rates = np.asarray([IXX * p, IYY * q, IZZ * r], dtype=np.float64)
    gyro = np.cross(np.asarray([p, q, r], dtype=np.float64), inertia_rates)
    angular_acc = (moment_body - gyro) / np.asarray([IXX, IYY, IZZ], dtype=np.float64)

    return np.array(
        [
            d_dx, d_dy, d_dz,
            d_vx, d_vy, d_vz,
            d_phi, d_theta, d_psi,
            angular_acc[0], angular_acc[1], angular_acc[2],
            0.0, 0.0, 0.0,
            d_omega1, d_omega2, d_omega3, d_omega4,
        ],
        dtype=np.float64,
    )
