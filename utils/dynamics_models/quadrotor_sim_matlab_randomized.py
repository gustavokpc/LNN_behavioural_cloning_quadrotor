#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bebop2 aerodynamic dynamics translated from FM_BB2_6DOF.m.

The rotor yaw sign is adapted to the controller-training convention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import DynamicsInfo


G = 9.81
MASS = 0.510 # paper values
IXX = 0.001920 #paper values
IYY = 0.001850 #paper values
IZZ = 0.003340 #paper values
OMEGA_MAX = 10000.0
OMEGA_MIN = 5000.0
DEFAULT_TAU = 0.06
TAU = DEFAULT_TAU

RHO = 1.225
R = 0.075
L = 0.088
B = 0.115
AREA = np.pi * R**2
S = 4.0 * B * L
RPM_TO_RAD_S = 2.0 * np.pi / 60.0

SIGNR = -1.0
SL = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
SM = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
SN = SIGNR * np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
ROTOR_POSITIONS = np.array(
    [
        [L, -B, 0.0],
        [L, B, 0.0],
        [-L, B, 0.0],
        [-L, -B, 0.0],
    ],
    dtype=np.float64,
)

K_CT0 = np.array(
    [
        1.5608699335679425e-02,
        -5.5203712207002993e-02,
        6.8373739471780537e-01,
        -2.2386746543744325e00,
        3.0487551954496670e00,
        -1.5151394628763879e00,
        -1.4452997585238964e-02,
        4.5664172755604832e-01,
        -5.2451229104942632e-01,
        2.3311292320225827e-01,
        -2.5847908175511535e-02,
        4.0081035350267295e-02,
        -1.1555207086648811e-02,
        -2.2312195771809176e-03,
        -2.2536635249275513e-02,
        3.3603579143813935e-03,
    ],
    dtype=np.float64,
)
K_CQ0 = np.array(
    [
        -2.2666283737348665e-03,
        -1.1272325229483371e-03,
        3.6755812531864369e-03,
        -1.0084483009293548e-01,
        2.2598950398682635e-01,
        -1.4625704645524937e-01,
        -3.0488011051989740e-03,
        -7.4756001696528445e-03,
        -1.1112951725382153e-01,
        1.2154218606659822e-01,
        3.3562635661743183e-03,
        3.6340289707626175e-03,
        -7.2880066350763410e-03,
        1.1629568968266250e-03,
        2.5734357204199219e-03,
        -6.8056363992066531e-04,
    ],
    dtype=np.float64,
)
K_MODEL = {
    1: np.array([8.3830000000000005e-01, -2.8536999999999999e00], dtype=np.float64),
    2: np.array([3.0010201085253096e-02, -8.9189122199398146e-02], dtype=np.float64),
    3: np.array([-5.0932092901974557e-01, 3.9966451853283669e-01], dtype=np.float64),
    4: -3.9621480670528505e-05,
    5: 2.2925907776961882e-05,
    6: 4.6444946965533518e-06,
    7: -9.6610559042848528e-07,
    8: np.array(
        [
            1.1744348774819557e-02,
            -4.9783056660455006e-04,
            1.6876894655524964e-02,
            -2.6481727161199176e-02,
            6.2434913619265037e-02,
            -4.5688818395152909e-02,
        ],
        dtype=np.float64,
    ),
    9: np.array(
        [
            -8.4625851462271237e-02,
            2.7646286183395224e-01,
            1.0188252424024791e-01,
            -1.9418572213526053e-01,
            3.6284183800335434e-02,
            1.9786776362575691e-02,
        ],
        dtype=np.float64,
    ),
    10: np.array(
        [
            -3.0729246620140552e-02,
            7.5908541159230708e-02,
            -1.3399020881700217e-02,
            -4.5066769504689422e-02,
            1.5521950307307686e-02,
            -5.7219661069848118e-03,
        ],
        dtype=np.float64,
    ),
}


@dataclass(frozen=True)
class DynamicsParameters:
    """Physical and aerodynamic parameters for one simulated vehicle."""

    mass: float = MASS
    ixx: float = IXX
    iyy: float = IYY
    izz: float = IZZ
    omega_min: float = OMEGA_MIN
    omega_max: float = OMEGA_MAX
    tau: float = DEFAULT_TAU
    rho: float = RHO
    rotor_radius: float = R
    arm_x: float = L
    arm_y: float = B
    k_ct0: np.ndarray = None
    k_cq0: np.ndarray = None
    k_model: dict[int, np.ndarray | float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_ct0", K_CT0.copy() if self.k_ct0 is None else np.asarray(self.k_ct0, dtype=np.float64).copy())
        object.__setattr__(self, "k_cq0", K_CQ0.copy() if self.k_cq0 is None else np.asarray(self.k_cq0, dtype=np.float64).copy())
        model = K_MODEL if self.k_model is None else self.k_model
        object.__setattr__(
            self,
            "k_model",
            {key: value.copy() if isinstance(value, np.ndarray) else float(value) for key, value in model.items()},
        )
        for name in ("mass", "ixx", "iyy", "izz", "omega_max", "tau", "omega_min", "rotor_radius", "arm_x", "arm_y"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")
        if self.omega_max <= self.omega_min:
            raise ValueError("omega_max must be greater than omega_min.")

    @property
    def area(self) -> float:
        return float(np.pi * self.rotor_radius**2)

    @property
    def reference_area(self) -> float:
        return float(4.0 * self.arm_y * self.arm_x)

    @property
    def rotor_positions(self) -> np.ndarray:
        return np.asarray(
            [[self.arm_x, -self.arm_y, 0.0], [self.arm_x, self.arm_y, 0.0],
             [-self.arm_x, self.arm_y, 0.0], [-self.arm_x, -self.arm_y, 0.0]],
            dtype=np.float64,
        )


DEFAULT_PARAMETERS = DynamicsParameters()


def sample_randomized_parameters(
    randomization_factor: float = 0.30,
    *,
    rng: np.random.Generator | int | None = None,
    base: DynamicsParameters = DEFAULT_PARAMETERS,
    randomize_aerodynamic_coefficients: bool = True,
) -> DynamicsParameters:
    """Sample one parameter set uniformly within ``±randomization_factor``.

    A factor of ``0.30`` matches the ±30% uniform randomization pattern used
    in ``quadcopter_hover_envs.py``. Pass ``0.0`` for fixed parameters.
    """
    factor = float(randomization_factor)
    if not 0.0 <= factor < 1.0:
        raise ValueError(f"randomization_factor must be in [0, 1), got {factor}.")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    def scale(value):
        value_array = np.asarray(value, dtype=np.float64)
        sampled = value_array * generator.uniform(1.0 - factor, 1.0 + factor, size=value_array.shape)
        return float(sampled) if sampled.ndim == 0 else sampled

    k_model = {
        key: scale(value) if randomize_aerodynamic_coefficients else (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in base.k_model.items()
    }
    sampled_omega_min = max(0.0, scale(base.omega_min))
    sampled_omega_max = max(scale(base.omega_max), sampled_omega_min + 1.0)
    return DynamicsParameters(
        mass=scale(base.mass),
        ixx=scale(base.ixx),
        iyy=scale(base.iyy),
        izz=scale(base.izz),
        omega_min=sampled_omega_min,
        omega_max=sampled_omega_max,
        tau=scale(base.tau),
        rho=scale(base.rho),
        rotor_radius=scale(base.rotor_radius),
        arm_x=scale(base.arm_x),
        arm_y=scale(base.arm_y),
        k_ct0=scale(base.k_ct0) if randomize_aerodynamic_coefficients else base.k_ct0,
        k_cq0=scale(base.k_cq0) if randomize_aerodynamic_coefficients else base.k_cq0,
        k_model=k_model,
    )


def fixed_parameters() -> DynamicsParameters:
    return DEFAULT_PARAMETERS


def randomization_10_percent(*, rng=None) -> DynamicsParameters:
    return sample_randomized_parameters(0.10, rng=rng)


def randomization_20_percent(*, rng=None) -> DynamicsParameters:
    return sample_randomized_parameters(0.20, rng=rng)


def randomization_30_percent(*, rng=None) -> DynamicsParameters:
    return sample_randomized_parameters(0.30, rng=rng)

INFO = DynamicsInfo(
    name="quadrotor_sim_matlab",
    description="Bebop2 Matlab force/moment model with rotor yaw sign adapted to the controller-training convention.",
    omega_min=OMEGA_MIN,
    omega_max=OMEGA_MAX,
    tau=TAU,
    hover_omega=(G / (4.0 * (K_CT0[0] * RHO * R**2 * AREA / MASS) * RPM_TO_RAD_S**2)) ** 0.5,
)


def set_motor_tau(tau: float) -> None:
    """Set the motor-response time constant used by subsequent integrations."""
    global TAU, INFO, DEFAULT_PARAMETERS
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError(f"Motor tau must be positive, got {tau}.")
    TAU = tau
    INFO = replace(INFO, tau=tau)
    DEFAULT_PARAMETERS = replace(DEFAULT_PARAMETERS, tau=tau)


def _safe_sign(value: float) -> float:
    return 0.0 if value == 0.0 else float(np.sign(value))


def _p1n(x1: float, n: int, bias: bool, u: float) -> np.ndarray:
    terms: list[float] = []
    if bias:
        terms.append(u)
    terms.extend((x1**power) * u for power in range(1, n + 1))
    return np.asarray(terms, dtype=np.float64)


def _p32(x1: float, x2: float, u: float, bias: bool = False) -> np.ndarray:
    terms: list[float] = []
    if bias:
        terms.append(u)
    terms.extend(
        [
            x1 * u,
            x1**2 * u,  
            x1 * x2 * u,
            x1**3 * u,
            x1 * x2**2 * u,
            x1**2 * x2 * u,
        ]
    )
    return np.asarray(terms, dtype=np.float64)


def _p52_ct_cq(alpha: float, mu: float) -> np.ndarray:
    return np.asarray(
        [
            1.0,
            mu,
            mu**2,
            mu**3,
            mu**4,
            mu**5,
            mu * alpha,
            mu**2 * alpha,
            mu**3 * alpha,
            mu**4 * alpha,
            mu * alpha**2,
            mu**2 * alpha**2,
            mu**3 * alpha**2,
            mu * alpha**3,
            mu**2 * alpha**3,
            mu * alpha**4,
        ],
        dtype=np.float64,
    )


def forces_moments(
    vel: np.ndarray,
    rates: np.ndarray,
    omega_rpm: np.ndarray,
    rotor_yaw_sign: float = 1.0, #was -1.0 for bebop1
    parameters: DynamicsParameters | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    params = DEFAULT_PARAMETERS if parameters is None else parameters
    vel = np.asarray(vel, dtype=np.float64).reshape(3)
    rates = np.asarray(rates, dtype=np.float64).reshape(3)
    omega = np.asarray(omega_rpm, dtype=np.float64).reshape(4) * RPM_TO_RAD_S
    omega = np.where(omega <= 1.0, 0.0, omega)

    u, v, w = vel
    va = float(np.linalg.norm(vel))
    if va >= 0.01:
        u_bar, v_bar, w_bar = vel / va
    else:
        u_bar = v_bar = w_bar = 0.0

    rotor_vel = np.asarray([np.cross(rates, arm) + vel for arm in params.rotor_positions], dtype=np.float64)
    rotor_speed = np.linalg.norm(rotor_vel, axis=1)

    mu = np.zeros(4, dtype=np.float64)
    nonzero = omega != 0.0
    mu[nonzero] = rotor_speed[nonzero] / (omega[nonzero] * params.rotor_radius)
    mu = np.clip(np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 0.6)

    alpha = np.zeros(4, dtype=np.float64)
    xy_speed = np.linalg.norm(rotor_vel[:, 0:2], axis=1)
    valid_alpha = xy_speed != 0.0
    alpha[valid_alpha] = np.arctan(rotor_vel[valid_alpha, 2] / xy_speed[valid_alpha])
    alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

    dynhead = params.rho * omega**2 * params.rotor_radius**2
    ct = np.asarray([_p52_ct_cq(alpha_i, mu_i) @ params.k_ct0 for alpha_i, mu_i in zip(alpha, mu)], dtype=np.float64)
    cq = np.asarray([_p52_ct_cq(alpha_i, mu_i) @ params.k_cq0 for alpha_i, mu_i in zip(alpha, mu)], dtype=np.float64)

    qbar_s = va**2 * params.reference_area * params.rho / 2.0
    t0 = -float(_p1n(abs(w_bar), 2, False, qbar_s * _safe_sign(w_bar)) @ params.k_model[1])
    x0 = float(_p1n(abs(u_bar), 2, False, qbar_s * _safe_sign(u_bar)) @ params.k_model[2])
    y0 = float(_p1n(abs(v_bar), 2, False, qbar_s * _safe_sign(v_bar)) @ params.k_model[3])
    l0 = float(_p32(abs(v_bar), w_bar, qbar_s * _safe_sign(v_bar)) @ params.k_model[8])
    m0 = float(_p32(abs(u_bar), w_bar, qbar_s * _safe_sign(u_bar)) @ params.k_model[9])
    n0 = float(_p32(abs(v_bar), abs(u_bar), qbar_s * _safe_sign(v_bar) * _safe_sign(u_bar)) @ params.k_model[10])

    thrust = ct * dynhead * params.area
    rotor_u = rotor_vel[:, 0]
    rotor_v = rotor_vel[:, 1]
    x_forces = rotor_u * omega * params.k_model[4] + SN * rotor_v * omega * params.k_model[5]
    y_forces = -SN * rotor_u * omega * params.k_model[5] + rotor_v * omega * params.k_model[4]
    l_moments = SN * rotor_u * omega * params.k_model[7] - rotor_v * omega * params.k_model[6]
    m_moments = rotor_u * omega * params.k_model[6] + SN * rotor_v * omega * params.k_model[7]
    n_moments = rotor_yaw_sign * SN * cq * dynhead * params.area * params.rotor_radius

    total_thrust = float(thrust.sum() + t0)
    fx = float(x_forces.sum() + x0)
    fy = float(y_forces.sum() + y0)
    mx = float((SL * params.arm_y * thrust).sum() + l0 + l_moments.sum())
    my = float((SM * params.arm_x * thrust).sum() + m0 + m_moments.sum())
    mz = float((params.arm_y * SL * x_forces).sum() + (params.arm_x * SM * y_forces).sum() + n_moments.sum() + n0)

    return np.asarray([fx, fy, -total_thrust], dtype=np.float64), np.asarray([mx, my, mz], dtype=np.float64)


def dynamics(
    state: np.ndarray,
    action: np.ndarray,
    parameters: DynamicsParameters | None = None,
    rotor_yaw_sign: float = 1.0,
) -> np.ndarray:
    """Compute state derivatives using fixed or sampled vehicle parameters."""
    params = DEFAULT_PARAMETERS if parameters is None else parameters
    (
        dx, dy, dz, vx, vy, vz,
        phi, theta, psi, p, q, r,
        mx_ext, my_ext, mz_ext, omega1, omega2, omega3, omega4,
    ) = state
    u1, u2, u3, u4 = action

    d_dx = -q * dz + r * dy - vx
    d_dy = p * dz - r * dx - vy
    d_dz = -p * dy + q * dx - vz

    d_omega1 = (params.omega_min + u1 * (params.omega_max - params.omega_min) - omega1) / params.tau
    d_omega2 = (params.omega_min + u2 * (params.omega_max - params.omega_min) - omega2) / params.tau
    d_omega3 = (params.omega_min + u3 * (params.omega_max - params.omega_min) - omega3) / params.tau
    d_omega4 = (params.omega_min + u4 * (params.omega_max - params.omega_min) - omega4) / params.tau

    force_body, moment_body = forces_moments(
        vel=np.asarray([vx, vy, vz], dtype=np.float64),
        rates=np.asarray([p, q, r], dtype=np.float64),
        omega_rpm=np.asarray([omega1, omega2, omega3, omega4], dtype=np.float64),
        rotor_yaw_sign=rotor_yaw_sign,
        parameters=params,
    )
    moment_body = moment_body + np.asarray([mx_ext, my_ext, mz_ext], dtype=np.float64)

    d_vx = -q * vz + r * vy - G * np.sin(theta) + force_body[0] / params.mass
    d_vy = p * vz - r * vx + G * np.cos(theta) * np.sin(phi) + force_body[1] / params.mass
    d_vz = -p * vy + q * vx + G * np.cos(theta) * np.cos(phi) + force_body[2] / params.mass

    d_phi = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
    d_theta = q * np.cos(phi) - r * np.sin(phi)
    d_psi = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)

    inertia_rates = np.asarray([params.ixx * p, params.iyy * q, params.izz * r], dtype=np.float64)
    gyro = np.cross(np.asarray([p, q, r], dtype=np.float64), inertia_rates)
    angular_acc = (moment_body - gyro) / np.asarray([params.ixx, params.iyy, params.izz], dtype=np.float64)

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
