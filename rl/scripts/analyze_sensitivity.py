"""Instantaneous Jacobians, learned local gains, and finite-history saliency."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..analysis.io import write_json
from ..analysis.plotting import plot_heatmap
from ..analysis.sensitivity import history_saliency, instantaneous_sensitivity, virtual_control_gain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    instant = commands.add_parser("instantaneous")
    instant.add_argument("--result", required=True)
    instant.add_argument("--checkpoint")
    instant.add_argument("--architecture", default="auto")
    instant.add_argument("--device", default="auto")
    instant.add_argument("--samples", type=int, default=64)
    instant.add_argument("--event", choices=("gate_crossing", "dropout_onset", "observation_restoration", "perturbation_onset"))
    instant.add_argument("--phase", choices=("all", "stabilized", "gate_approach", "turning", "recovery"), default="all")
    instant.add_argument("--virtual-controls", action="store_true")

    history = commands.add_parser("history")
    history.add_argument("--result", required=True)
    history.add_argument("--checkpoint")
    history.add_argument("--architecture", default="auto")
    history.add_argument("--device", default="auto")
    history.add_argument("--history", type=int, default=50)
    history.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    result_dir = Path(args.result)
    if args.command == "instantaneous":
        result = instantaneous_sensitivity(
            result_dir, args.checkpoint, architecture=args.architecture, device=args.device,
            samples=args.samples, event=args.event, phase=args.phase,
        )
        matrix = np.asarray(result["mean_absolute_jacobian"])
        plot_heatmap(
            matrix, [f"obs_{i}" for i in range(matrix.shape[1])], [f"motor_{i + 1}" for i in range(matrix.shape[0])],
            result_dir / "figures" / "instantaneous_sensitivity.png", "E[|du_i/dx_j|]",
        )
        if args.virtual_controls:
            virtual = virtual_control_gain(np.asarray(result["mean_jacobian"]))
            write_json(result_dir / "virtual_control_gain.json", virtual)
            plot_heatmap(
                np.asarray(virtual["gain"]), [f"obs_{i}" for i in range(matrix.shape[1])], virtual["labels"],
                result_dir / "figures" / "virtual_control_gain.png", "Approximate local learned gain",
            )
        print(result_dir / "instantaneous_sensitivity.json")
    else:
        result = history_saliency(
            result_dir, args.checkpoint, architecture=args.architecture, device=args.device,
            history=args.history, samples=args.samples,
        )
        matrix = np.asarray(result["lag_observation_saliency"])
        plot_heatmap(
            matrix, [f"obs_{i}" for i in range(matrix.shape[1])], [str(i) for i in range(matrix.shape[0])],
            result_dir / "figures" / "history_saliency.png", "Lag × observation saliency",
        )
        print(result_dir / "history_saliency.json")


if __name__ == "__main__":
    main()
