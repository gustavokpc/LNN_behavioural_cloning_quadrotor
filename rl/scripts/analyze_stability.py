"""Exploratory sample-local recurrent and closed-loop stability diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..analysis.io import write_json
from ..analysis.plotting import plot_stability_summary
from ..analysis.stability import (
    analyze_closed_loop_stability,
    analyze_recurrent_stability,
    common_quadratic_lyapunov,
    finite_time_contraction,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recurrent = commands.add_parser("recurrent")
    recurrent.add_argument("--result", required=True)
    recurrent.add_argument("--sample-index", type=int, default=0)
    recurrent.add_argument("--device", default="cpu")
    closed = commands.add_parser("closed-loop")
    closed.add_argument("--result", required=True)
    closed.add_argument("--sample-index", type=int, default=0)
    closed.add_argument("--epsilon", type=float, nargs="+", default=(1e-4, 1e-5))
    closed.add_argument("--device", default="cpu")
    closed.add_argument("--window", type=int, default=1, help="Consecutive saved points for finite-time Jacobian products.")
    lmi = commands.add_parser("lmi")
    lmi.add_argument("--jacobians", required=True, help="NPZ containing arrays whose names start with jacobian.")
    lmi.add_argument("--output", required=True)
    lmi.add_argument("--margin", type=float, default=1e-6)
    args = parser.parse_args()
    if args.command == "recurrent":
        analyze_recurrent_stability(args.result, sample_index=args.sample_index, device=args.device)
        plot_stability_summary(Path(args.result) / "recurrent_stability.json", Path(args.result) / "figures" / "recurrent_stability.png")
        print(Path(args.result) / "recurrent_stability.json")
    elif args.command == "closed-loop":
        result_path = Path(args.result)
        if args.window <= 1:
            analyze_closed_loop_stability(
                args.result, sample_index=args.sample_index, epsilon_values=args.epsilon, device=args.device
            )
            plot_stability_summary(result_path / "closed_loop_stability.json", result_path / "figures" / "closed_loop_stability.png")
            print(result_path / "closed_loop_stability.json")
        else:
            jacobians = []
            point_results = []
            for index in range(args.sample_index, args.sample_index + args.window):
                point_results.append(analyze_closed_loop_stability(
                    args.result, sample_index=index, epsilon_values=[args.epsilon[0]], device=args.device
                ))
                with np.load(result_path / "closed_loop_stability.npz") as data:
                    jacobians.append(data["jacobian_0"])
            from ..analysis.io import read_json

            dt = float(read_json(result_path / "metadata.json")["dt"])
            window_result = {
                "scope": "finite product of consecutive sample-local closed-loop Jacobians; windows crossing switches are rejected",
                "sample_indices": list(range(args.sample_index, args.sample_index + args.window)),
                "epsilon": args.epsilon[0],
                "points": point_results,
                "finite_time": finite_time_contraction(jacobians, dt),
            }
            np.savez_compressed(result_path / "closed_loop_window.npz", **{f"jacobian_{i}": value for i, value in enumerate(jacobians)})
            write_json(result_path / "closed_loop_window.json", window_result)
            print(result_path / "closed_loop_window.json")
    else:
        with np.load(args.jacobians) as data:
            matrices = [data[name] for name in data.files if name.startswith("jacobian")]
        result = common_quadratic_lyapunov(matrices, margin=args.margin)
        write_json(args.output, result)
        print(args.output)


if __name__ == "__main__":
    main()
