#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supaero/Bebop2-style quadrotor dynamics using only the provided parameters.

State layout is kept compatible with quadrotor_sim.py:
    [dx, dy, dz, vx, vy, vz, phi, theta, psi, p, q, r,
     mx_ext, my_ext, mz_ext, omega1, omega2, omega3, omega4]

Action layout is also kept compatible:
    [u1, u2, u3, u4], with ui normalized in [0, 1]

Important convention:
- omega_i is represented in RPM, as in the original Ferede/TUDelft simulator.
- The translational equations follow the same body-frame/NED convention used by
  quadrotor_sim.py: positive body z is down, therefore rotor thrust appears as a
  negative Fz force.
"""

from __future__ import annotations

import numpy as np

from . import DynamicsInfo


# -----------------------------------------------------------------------------
# Physical constants and rigid-body parameters
# -----------------------------------------------------------------------------
G = 9.81
MASS = 0.500
IXX = 0.001805
IYY = 0.001764
IZZ = 0.003328

OMEGA_MAX = 10000.0  # [RPM]
OMEGA_MIN = 5000.0   # [RPM]
TAU = 0.06           # [s]

# -----------------------------------------------------------------------------
# Native Supaero parameters provided by the simulator/model
# -----------------------------------------------------------------------------
# Linear damping coefficients [s^-1].  The model also had Shell values, but they
# are identical to NoShell in the data provided, so only one set is used here.
CX = 0.35
CY = 0.40
CZ = 0.90
CR = 0.50

EFFICIENCY_RPM_TO_ROLL = 0.10
EFFICIENCY_RPM_TO_PITCH = 0.078
EFFICIENCY_RPM_TO_YAW = 0.0053
EFFICIENCY_RPM_PER_SECOND_TO_YAW = 0.001
EFFICIENCY_RPM_TO_Z = 0.0032

MOTOR_OMEGA0 = 90.0
MOTOR_SIGMA = 0.9

# Motor thrust/drag polynomial parameters supplied with the model.
MOTOR_A = np.array([2.15e-8, 2.15e-8, 2.15e-8, 2.15e-8], dtype=np.float64)      # [N/RPM^2]
MOTOR_AL = np.array([3.93e-4, 3.93e-4, 3.93e-4, 3.93e-4], dtype=np.float64)     # [N/RPM]
MOTOR_B = np.array([-2.85e-10, 2.85e-10, -2.85e-10, 2.85e-10], dtype=np.float64) # [N.m/RPM^2]
MOTOR_BL = np.array([-4.41e-6, 4.41e-6, -4.41e-6, 4.41e-6], dtype=np.float64)   # [N.m/RPM]
MOTOR_JP = np.array([7.94e-6, 7.94e-6, 7.94e-6, 7.94e-6], dtype=np.float64)     # [kg.m^2]
MOTOR_ALPHA = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
MOTOR_BETA = np.array(
    [-1.5707963267948966, -1.5707963267948966, -1.5707963267948966, -1.5707963267948966],
    dtype=np.float64,
)

# Rotor positions in body frame [m].
MOTOR_X = np.array([0.0875, 0.0875, -0.0875, -0.0875], dtype=np.float64)
MOTOR_Y = np.array([-0.115, 0.115, 0.115, -0.115], dtype=np.float64)
MOTOR_Z = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
ROTOR_POSITIONS = np.column_stack((MOTOR_X, MOTOR_Y, MOTOR_Z))
RPM_ERROR_MARGIN = 1.0

RPM_TO_RAD_S = 2.0 * np.pi / 60.0
EFFICIENCY_RPM_SCALE = 10.0


def _hover_omega() -> float:
    """Return equal-motor hover RPM from the Supaero efficiency model."""
    return G * EFFICIENCY_RPM_SCALE / (4.0 * EFFICIENCY_RPM_TO_Z)


INFO = DynamicsInfo(
    name="quadrotor_sim_supaero",
    description="Supaero native quadrotor dynamics using only the provided native parameters.",
    omega_min=OMEGA_MIN,
    omega_max=OMEGA_MAX,
    tau=TAU,
    hover_omega=_hover_omega(),
)


def _scaled_rpm(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(4) / EFFICIENCY_RPM_SCALE


def dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    (
        dx, dy, dz, vx, vy, vz,
        phi, theta, psi, p, q, r,
        mx_ext, my_ext, mz_ext, omega1, omega2, omega3, omega4,
    ) = np.asarray(state, dtype=np.float64)

    u1, u2, u3, u4 = np.asarray(action, dtype=np.float64)
    # Keep commands inside the physical normalized range.
    u = np.clip(np.array([u1, u2, u3, u4], dtype=np.float64), 0.0, 1.0)
    omega = np.array([omega1, omega2, omega3, omega4], dtype=np.float64)

    # Relative-position dynamics, same as quadrotor_sim.py.
    d_dx = -q * dz + r * dy - vx
    d_dy = p * dz - r * dx - vy
    d_dz = -p * dy + q * dx - vz

    # First-order motor dynamics.
    omega_cmd = OMEGA_MIN + u * (OMEGA_MAX - OMEGA_MIN)
    d_omega = (omega_cmd - omega) / TAU

    omega_eff = _scaled_rpm(omega)
    d_omega_eff = _scaled_rpm(d_omega)
    roll_mix = omega_eff[0] - omega_eff[1] - omega_eff[2] + omega_eff[3]
    pitch_mix = omega_eff[0] + omega_eff[1] - omega_eff[2] - omega_eff[3]
    yaw_mix = -omega_eff[0] + omega_eff[1] - omega_eff[2] + omega_eff[3]
    yaw_dot_mix = -d_omega_eff[0] + d_omega_eff[1] - d_omega_eff[2] + d_omega_eff[3]

    # Translational dynamics in body-frame form, matching quadrotor_sim.py.
    # CX/CY/CZ are native linear damping accelerations [s^-1].
    thrust_acc = EFFICIENCY_RPM_TO_Z * float(np.sum(omega_eff))
    d_vx = -q * vz + r * vy - G * np.sin(theta) - CX * vx
    d_vy = p * vz - r * vx + G * np.cos(theta) * np.sin(phi) - CY * vy
    d_vz = -p * vy + q * vx + G * np.cos(theta) * np.cos(phi) - thrust_acc - CZ * vz

    # Euler angle kinematics.
    cos_theta = np.cos(theta)
    if abs(cos_theta) < 1e-6:
        cos_theta = np.sign(cos_theta) * 1e-6 if cos_theta != 0.0 else 1e-6
    d_phi = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
    d_theta = q * np.cos(phi) - r * np.sin(phi)
    d_psi = q * np.sin(phi) / cos_theta + r * np.cos(phi) / cos_theta

    d_p = (
        q * r * (IYY - IZZ) / IXX
        + EFFICIENCY_RPM_TO_ROLL * roll_mix
        + mx_ext / IXX
    )
    d_q = (
        p * r * (IZZ - IXX) / IYY
        + EFFICIENCY_RPM_TO_PITCH * pitch_mix
        + my_ext / IYY
    )
    d_r = (
        p * q * (IXX - IYY) / IZZ
        + EFFICIENCY_RPM_TO_YAW * yaw_mix
        + EFFICIENCY_RPM_PER_SECOND_TO_YAW * yaw_dot_mix
        - CR * r
        + mz_ext / IZZ
    )

    return np.array(
        [
            d_dx, d_dy, d_dz,
            d_vx, d_vy, d_vz,
            d_phi, d_theta, d_psi,
            d_p, d_q, d_r,
            0.0, 0.0, 0.0,
            d_omega[0], d_omega[1], d_omega[2], d_omega[3],
        ],
        dtype=np.float64,
    )
