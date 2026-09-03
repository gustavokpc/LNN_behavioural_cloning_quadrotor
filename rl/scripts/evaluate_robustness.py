"""Generic sweeps over existing robustness hooks and evaluation-only interventions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..analysis.io import flatten_summary_means, read_json, write_json, write_rows_csv
from ..analysis.loading import build_args
from ..analysis.plotting import plot_robustness
from ..analysis.robustness import first_degradation_level, normalized_robustness_auc, robustness_degradation
from ..analysis.recovery import recovery_metrics
from ..analysis.rollout import (
    DropoutConfig,
    InitialConditionConfig,
    PerturbationConfig,
    RolloutConfig,
    collect_rollouts,
)
from ._common import add_environment_arguments, add_model_arguments, environment_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=(
        "initial_condition", "observation_noise", "rate_noise", "dynamics", "aerodynamics",
        "actuator_tau", "dt_matched", "dt_fixed", "dropout_velocity", "dropout_angular_rate", "perturbation",
    ))
    parser.add_argument("--stress-values", type=float, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--record-recurrent", action="store_true")
    parser.add_argument("--dropout-onset", type=float, default=1.0)
    parser.add_argument("--dropout-onset-event", choices=("time", "first_gate_crossing", "perturbation_onset"), default="time")
    parser.add_argument("--dropout-mode", choices=("zero", "freeze"), default="zero")
    parser.add_argument("--perturbation-onset", type=float, default=1.0)
    parser.add_argument("--perturbation-kind", choices=("position", "velocity", "attitude", "angular_rate", "external_moment"), default="velocity")
    parser.add_argument("--perturbation-vector", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    parser.add_argument("--degradation-threshold", type=float, default=20.0)
    add_environment_arguments(parser)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_overrides = environment_overrides(args)
    base_args, _ = build_args(args.checkpoint, args.architecture, base_overrides, args.device)
    nominal_dt = float(base_args.dt)
    nominal_max_steps = int(base_args.max_steps)
    rows = []
    for index, stress in enumerate(args.stress_values):
        overrides = dict(base_overrides)
        initial = InitialConditionConfig()
        dropout = DropoutConfig()
        perturbation = PerturbationConfig()
        policy_dt = None
        if args.mode == "initial_condition":
            initial = InitialConditionConfig(stress, stress, stress, stress)
        elif args.mode == "observation_noise":
            overrides["param_input_noise"] = stress
        elif args.mode == "rate_noise":
            overrides["obs_rate_noise_std"] = (stress, stress, stress)
        elif args.mode in {"dynamics", "aerodynamics"}:
            overrides.update(randomize_dynamics=stress > 0.0, randomization_factor=stress)
            overrides["randomize_aerodynamic_coefficients"] = args.mode == "aerodynamics"
        elif args.mode == "actuator_tau":
            overrides.update(motor_tau=stress, tau=stress)
        elif args.mode in {"dt_matched", "dt_fixed"}:
            overrides["dt"] = stress
            overrides["max_steps"] = max(1, int(round(nominal_max_steps * nominal_dt / stress)))
            policy_dt = stress if args.mode == "dt_matched" else nominal_dt
        elif args.mode.startswith("dropout_"):
            dropout = DropoutConfig(
                kind=args.mode[len("dropout_"):], mode=args.dropout_mode,
                onset_s=args.dropout_onset, duration_s=stress, onset_event=args.dropout_onset_event,
            )
        else:
            perturbation = PerturbationConfig(
                onset_s=args.perturbation_onset,
                values={args.perturbation_kind: (np.asarray(args.perturbation_vector) * stress).tolist()},
            )
        result_dir = output / f"stress_{index:03d}_{stress:g}"
        collect_rollouts(RolloutConfig(
            checkpoint=args.checkpoint, output_dir=str(result_dir), architecture=args.architecture,
            label=f"{args.mode}_{stress:g}", episodes=args.episodes, seed=args.seed, device=args.device,
            record_recurrent=args.record_recurrent, record_internals=args.record_recurrent,
            env_overrides=overrides, policy_dt=policy_dt, dropout=dropout,
            initial_condition=initial, perturbation=perturbation,
        ))
        if args.mode.startswith("dropout_"):
            recovery_metrics(result_dir, event="observation_restoration")
        elif args.mode == "perturbation":
            recovery_metrics(result_dir, event="perturbation_onset")
        means = flatten_summary_means(read_json(result_dir / "summary.json"))
        rows.append({"stress_value": stress, "result_dir": str(result_dir), **means})
    important = [metric for metric in ("total_reward", "gates_crossed", "full_cycle_success", "success", "failure", "policy_action_delta_rms") if metric in rows[0]]
    nominal = rows[0]
    enriched = []
    for row in rows:
        enriched.append({**row, "degradation": robustness_degradation(nominal, row, important)})
    diagnostics = {
        metric: {
            "normalized_auc": normalized_robustness_auc(args.stress_values, [row[metric] for row in rows]),
            "first_threshold_level": first_degradation_level(
                args.stress_values, [row[metric] for row in rows], args.degradation_threshold,
                higher_is_better=metric not in {"failure", "policy_action_delta_rms"},
            ),
        }
        for metric in important
    }
    result = {
        "mode": args.mode,
        "nominal_definition": "first supplied stress value",
        "rows": enriched,
        "diagnostics": diagnostics,
        "dt_mode": "matched policy dt" if args.mode == "dt_matched" else "fixed nominal policy dt" if args.mode == "dt_fixed" else None,
    }
    write_json(output / "robustness.json", result)
    write_rows_csv(output / "robustness.csv", rows)
    for metric in important:
        plot_robustness(rows, metric, output / "figures" / f"{metric}.png")
    print(output)


if __name__ == "__main__":
    main()
