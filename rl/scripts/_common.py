"""Shared CLI argument helpers."""

from __future__ import annotations

import argparse


def add_model_arguments(parser: argparse.ArgumentParser, checkpoint_required: bool = True, multiple: bool = False) -> None:
    parser.add_argument("--checkpoint", required=checkpoint_required, nargs="+" if multiple else None)
    parser.add_argument("--architecture", default="auto", choices=(
        "auto", "mlp", "ppo", "rnn", "ct_rnn", "cfc", "recurrent_ppo",
        "ltc", "recurrent_ppo_ltc", "ncp", "ncp_cfc", "recurrent_ppo_ncp_cfc",
    ))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)


def add_rollout_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="evaluation")
    parser.add_argument("--labels", nargs="*", help="Labels for multiple checkpoints (same order as --checkpoint).")
    parser.add_argument("--record-recurrent", action="store_true")
    parser.add_argument("--record-internals", action="store_true")


def add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--track", choices=("figure8_gates", "square_waypoints", "point_to_point"), default=argparse.SUPPRESS)
    parser.add_argument("--dt", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--motor-tau", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--tau", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--rotor-yaw-sign", type=float, choices=(-1.0, 1.0), default=argparse.SUPPRESS)
    parser.add_argument(
        "--no-vel", dest="no_vel", action="store_true", default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--with-vel", "--no-no-vel", dest="no_vel", action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-ang-vel", dest="no_ang_vel", action="store_true", default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--with-ang-vel", "--no-no-ang-vel", dest="no_ang_vel", action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--low-obs", dest="low_obs", action="store_true", default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--full-obs", "--no-low-obs", dest="low_obs", action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--param-input-noise", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--obs-rate-noise-std", type=float, nargs=3, default=argparse.SUPPRESS)
    parser.add_argument("--randomize-dynamics", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--randomization-factor", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--randomize-aerodynamic-coefficients", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--randomize-external-moments", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)


def environment_overrides(namespace: argparse.Namespace) -> dict:
    names = (
        "track", "dt", "max_steps", "motor_tau", "tau", "rotor_yaw_sign", "no_vel", "no_ang_vel", "low_obs",
        "param_input_noise", "obs_rate_noise_std", "randomize_dynamics", "randomization_factor",
        "randomize_aerodynamic_coefficients", "randomize_external_moments",
    )
    return {name: getattr(namespace, name) for name in names if hasattr(namespace, name)}
