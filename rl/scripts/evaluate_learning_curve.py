"""Evaluate existing checkpoints versus recovered training environment steps."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..analysis.learning import discover_checkpoints, evaluate_learning_curve
from ..analysis.plotting import plot_learning_curve
from ._common import add_environment_arguments, environment_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--discover-from")
    parser.add_argument("--architecture", default="auto")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    add_environment_arguments(parser)
    args = parser.parse_args()
    checkpoints = list(args.checkpoints or [])
    if args.discover_from:
        checkpoints.extend(str(path) for path in discover_checkpoints(args.discover_from))
    if not checkpoints:
        parser.error("Provide --checkpoints or --discover-from.")
    evaluate_learning_curve(
        checkpoints, args.output, architecture=args.architecture, episodes=args.episodes,
        seed=args.seed, device=args.device, success_threshold=args.success_threshold,
        env_overrides=environment_overrides(args),
    )
    plot_learning_curve(Path(args.output) / "learning_curve.json", "full_cycle_success", Path(args.output) / "figures" / "success_vs_steps.png")
    print(args.output)


if __name__ == "__main__":
    main()
