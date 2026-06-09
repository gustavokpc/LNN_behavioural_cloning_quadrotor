#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bebop 2 dynamics using the BebopS / RotorS parameters.

This file follows the same 19-state convention used by quadrotor_sim_matlab.py:

state = [dx, dy, dz,
         vx, vy, vz,
         phi, theta, psi,
         p, q, r,
         mx_ext, my_ext, mz_ext,
         omega1, omega2, omega3, omega4]

action = [u1, u2, u3, u4], with ui normally in [0, 1].

Important convention:
- The state motor speeds omega1..omega4 are stored in RPM, because this is what
  the original Python simulator/network pipeline expects.
- BebopS/RotorS motor constants use rad/s internally, so conversion is done only
  inside forces_moments().

This model is intentionally simpler than the FM_BB2_6DOF aerodynamic model:
it reproduces the RotorS/Gazebo-style mapping thrust = k_f * omega^2 and
reaction torque = k_m * thrust, with BebopS mass, inertia, rotor positions and
motor first-order dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from . import DynamicsInfo
except Exception:  # Allows running this file standalone for quick checks.
    @dataclass
    class DynamicsInfo:
        name: str
        description: str
        omega_min: float
        omega_max: float
        tau: float
        hover_omega: float | None = None


G = 9.81

# BebopS parameters from urdf/bebop_base.urdf.xacro
MASS = 0.5
IXX = 0.00389
IYY = 0.00389
IZZ = 0.0078
INERTIA = np.array([IXX, IYY, IZZ], dtype=np.float64)

# Rotor geometry. Coordinates are in the body frame, meters. The URDF rotor
# coordinates are not centered on the inertial origin used by this reduced
# rigid-body model, so moments must use arms relative to the rotor-plane center.
ROTOR_OFFSET_TOP = 0.03
ROTOR_POSITIONS_URDF = np.array(
    [
        [ 0.08440513, -0.09784210, ROTOR_OFFSET_TOP],  # 0 front_right, ccw
        [ 0.08440513,  0.09784210, ROTOR_OFFSET_TOP],  # 1 front_left, cw
        [-0.06853580,  0.09784210, ROTOR_OFFSET_TOP],  # 2 back_left, ccw
        [-0.06853580, -0.09784210, ROTOR_OFFSET_TOP],  # 3 back_right, cw
    ],
    dtype=np.float64,
)
ROTOR_POSITION_ORIGIN = np.array(
    [
        float(np.mean(ROTOR_POSITIONS_URDF[:, 0])),
        float(np.mean(ROTOR_POSITIONS_URDF[:, 1])),
        0.0,
    ],
    dtype=np.float64,
)
ROTOR_POSITIONS = ROTOR_POSITIONS_URDF - ROTOR_POSITION_ORIGIN

# BebopS motor/rotor parameters.
MOTOR_CONSTANT = 8.54858e-6           # thrust = k_f * omega_rad_s**2 [N]
MOMENT_CONSTANT = 0.016               # yaw torque = k_m * thrust [N*m]
ROTOR_DRAG_COEFFICIENT = 8.06428e-5
ROLLING_MOMENT_COEFFICIENT = 1.0e-6
ROTOR_VELOCITY_SLOWDOWN_SIM = 15.0    # Gazebo joint-velocity numerical trick, not used in physical force calc here.

# BebopS RotorS motor time constant.
TIME_CONSTANT_UP = 0.0125
TIME_CONSTANT_DOWN = 0.0125
TAU = TIME_CONSTANT_UP

RPM_TO_RAD_S = 2.0 * np.pi / 60.0
RAD_S_TO_RPM = 60.0 / (2.0 * np.pi)

# BebopS max_rot_velocity is in rad/s. The behavioural-cloning controller,
# however, was trained with motor-speed features and commands in [5000, 10000]
# RPM, and the exported C controller normalizes omega with those same limits.
PHYSICAL_OMEGA_MAX_RAD_S = 1475.0
PHYSICAL_OMEGA_MIN_RAD_S = 0.0
PHYSICAL_OMEGA_MAX = PHYSICAL_OMEGA_MAX_RAD_S * RAD_S_TO_RPM
PHYSICAL_OMEGA_MIN = PHYSICAL_OMEGA_MIN_RAD_S * RAD_S_TO_RPM
OMEGA_MIN = 5000.0
OMEGA_MAX = 10000.0

# Direction convention chosen to match the controller-training model's
# body-frame yaw signs: motor 1 and 3 increase r, motor 2 and 4 decrease it.
YAW_SIGN = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)

# Optional compatibility mode. Leave False to use the BebopS parameter literally.
# Set True for closed-loop tests with controllers trained on quadrotor_sim.py,
# whose learned hover is almost exactly u=0.5 in the 5000..10000 RPM command
# range.
USE_HOVER_RPM_COMPAT = True
HOVER_RPM_COMPAT = 7497.41181054513

# The exported controllers were trained on a reduced model with substantially
# higher angular authority than the literal BebopS mass/inertia/arm parameters.
# Keeping this compatibility scale avoids large transient attitudes that dump
# vertical thrust and make the closed-loop drone fall.
USE_MOMENT_COMPAT = True
MOMENT_COMPAT_SCALE = np.array([2.91776246, 2.34595856, 3.16631209], dtype=np.float64)

# Reduced-model angular acceleration terms used during controller training.
USE_TRAINING_DAMPING_COMPAT = True
TRAINING_IXX = 0.000906
TRAINING_IYY = 0.001242
TRAINING_IZZ = 0.002054
TRAINING_KPV = -0.00797102
TRAINING_KQV = 0.01292637
TRAINING_KRR = 0.00081293


def _motor_constant_used() -> float:
    if not USE_HOVER_RPM_COMPAT:
        return MOTOR_CONSTANT
    hover_rad_s = HOVER_RPM_COMPAT * RPM_TO_RAD_S
    return MASS * G / (4.0 * hover_rad_s**2)


K_F = _motor_constant_used()
HOVER_OMEGA = float(np.sqrt(MASS * G / (4.0 * K_F)) * RAD_S_TO_RPM)
U_HOVER = (HOVER_OMEGA - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN) if OMEGA_MAX > OMEGA_MIN else None

INFO = DynamicsInfo(
    name="quadrotor_sim_bebops",
    description=(
        "Bebop 2 RotorS/Gazebo-style dynamics using BebopS mass, inertia, "
        "rotor positions, motor constant, moment constant and motor time constants."
    ),
    omega_min=OMEGA_MIN,
    omega_max=OMEGA_MAX,
    tau=TAU,
    hover_omega=HOVER_OMEGA,
)


def forces_moments(
    vel: np.ndarray,
    rates: np.ndarray,
    omega_rpm: np.ndarray,
    include_rotor_drag: bool = True,
    include_rolling_moment: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return total body force [N] and body moment [N*m].

    Body-frame convention matches quadrotor_sim_matlab.py:
    - At level hover, thrust appears as a negative body-z force.
    - The translational dynamics add +G in body z, so hover requires Fz = -m*g.
    """
    vel = np.asarray(vel, dtype=np.float64).reshape(3)
    rates = np.asarray(rates, dtype=np.float64).reshape(3)
    omega_rpm = np.asarray(omega_rpm, dtype=np.float64).reshape(4)

    omega_rad_s = np.maximum(omega_rpm * RPM_TO_RAD_S, 0.0)
    thrust = K_F * omega_rad_s**2

    # Main vertical rotor forces. Positive thrust magnitude, force in -body-z.
    force_body = np.array([0.0, 0.0, -float(np.sum(thrust))], dtype=np.float64)

    # Arm moments. With force [0,0,-T], Mx = -y*T.  The pitch sign is chosen to
    # remain compatible with the previous quadrotor_sim_matlab.py convention.
    moments = np.zeros(3, dtype=np.float64)
    moments[0] = -float(np.sum(ROTOR_POSITIONS[:, 1] * thrust))
    moments[1] =  float(np.sum(ROTOR_POSITIONS[:, 0] * thrust))
    moments[2] =  float(np.sum(YAW_SIGN * MOMENT_CONSTANT * thrust))

    rotor_vel = np.asarray([np.cross(rates, arm) + vel for arm in ROTOR_POSITIONS], dtype=np.float64)

    if include_rotor_drag:
        # RotorS-style simplified in-plane drag. This is an approximation, but it
        # keeps the same parameter role as the BebopS Gazebo model.
        drag_xy = -ROTOR_DRAG_COEFFICIENT * omega_rad_s[:, None] * rotor_vel[:, 0:2]
        force_body[0] += float(np.sum(drag_xy[:, 0]))
        force_body[1] += float(np.sum(drag_xy[:, 1]))

        # Moments produced by applying those in-plane drag forces at rotor arms.
        for i in range(4):
            Fi = np.array([drag_xy[i, 0], drag_xy[i, 1], 0.0], dtype=np.float64)
            moments += np.cross(ROTOR_POSITIONS[i], Fi)

    if include_rolling_moment:
        # Small additional rolling/pitching moments used by RotorS-like models.
        # The sign follows damping: it opposes local in-plane velocity.
        rolling_xy = -ROLLING_MOMENT_COEFFICIENT * omega_rad_s[:, None] * rotor_vel[:, 0:2]
        moments[0] += float(np.sum(rolling_xy[:, 1]))
        moments[1] += float(np.sum(rolling_xy[:, 0]))

    if USE_MOMENT_COMPAT:
        moments *= MOMENT_COMPAT_SCALE

    return force_body, moments


def dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Compute state derivative for the 19-state body-frame simulator.

    action is normalized in [0, 1] and mapped to [OMEGA_MIN, OMEGA_MAX] RPM.
    """
    (
        dx, dy, dz,
        vx, vy, vz,
        phi, theta, psi,
        p, q, r,
        mx_ext, my_ext, mz_ext,
        omega1, omega2, omega3, omega4,
    ) = np.asarray(state, dtype=np.float64)

    u1, u2, u3, u4 = np.asarray(action, dtype=np.float64)
    action = np.clip(np.array([u1, u2, u3, u4], dtype=np.float64), 0.0, 1.0)
    omega = np.array([omega1, omega2, omega3, omega4], dtype=np.float64)
    omega_cmd = OMEGA_MIN + action * (OMEGA_MAX - OMEGA_MIN)

    tau = np.where(omega_cmd >= omega, TIME_CONSTANT_UP, TIME_CONSTANT_DOWN)
    d_omega = (omega_cmd - omega) / tau

    # Relative-position dynamics in body frame, same form as quadrotor_sim_matlab.py.
    d_dx = -q * dz + r * dy - vx
    d_dy =  p * dz - r * dx - vy
    d_dz = -p * dy + q * dx - vz

    force_body, moment_body = forces_moments(
        vel=np.array([vx, vy, vz], dtype=np.float64),
        rates=np.array([p, q, r], dtype=np.float64),
        omega_rpm=omega,
    )
    moment_body += np.array([mx_ext, my_ext, mz_ext], dtype=np.float64)

    d_vx = -q * vz + r * vy - G * np.sin(theta) + force_body[0] / MASS
    d_vy =  p * vz - r * vx + G * np.cos(theta) * np.sin(phi) + force_body[1] / MASS
    d_vz = -p * vy + q * vx + G * np.cos(theta) * np.cos(phi) + force_body[2] / MASS

    d_phi = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
    d_theta = q * np.cos(phi) - r * np.sin(phi)
    d_psi = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)

    rates_vec = np.array([p, q, r], dtype=np.float64)
    inertia_rates = INERTIA * rates_vec
    gyro = np.cross(rates_vec, inertia_rates)
    angular_acc = (moment_body - gyro) / INERTIA
    if USE_TRAINING_DAMPING_COMPAT:
        angular_acc += np.array(
            [
                TRAINING_KPV * vy / TRAINING_IXX,
                TRAINING_KQV * vx / TRAINING_IYY,
                -TRAINING_KRR * r / TRAINING_IZZ,
            ],
            dtype=np.float64,
        )

    return np.array(
        [
            d_dx, d_dy, d_dz,
            d_vx, d_vy, d_vz,
            d_phi, d_theta, d_psi,
            angular_acc[0], angular_acc[1], angular_acc[2],
            0.0, 0.0, 0.0,
            d_omega[0], d_omega[1], d_omega[2], d_omega[3],
        ],
        dtype=np.float64,
    )


if __name__ == "__main__":
    print(f"Model: {INFO.name}")
    print(f"omega_min = {OMEGA_MIN:.2f} RPM")
    print(f"omega_max = {OMEGA_MAX:.2f} RPM")
    print(f"hover_omega = {HOVER_OMEGA:.2f} RPM")
    print(f"u_hover = {U_HOVER:.4f}")
    print(f"K_F used = {K_F:.8e}")
