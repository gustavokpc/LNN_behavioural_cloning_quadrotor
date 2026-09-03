"""Nominal evaluation, seed aggregation, partial-observation comparison, and timing."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..analysis.efficiency import benchmark_inference
from ..analysis.io import flatten_summary_means, read_json, write_json
from ..analysis.linear_probe import compare_partial_observability
from ..analysis.plotting import plot_failure_breakdown, plot_latency_comparison, plot_metric_comparison
from ..analysis.rollout import RolloutConfig, collect_rollouts
from ..analysis.statistics import aggregate_training_seeds
from ._common import add_environment_arguments, add_model_arguments, add_rollout_arguments, environment_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    add_model_arguments(evaluate, multiple=True)
    add_rollout_arguments(evaluate)
    add_environment_arguments(evaluate)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("result_dirs", nargs="+")
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--metrics", nargs="*")

    partial = commands.add_parser("compare-partial")
    partial.add_argument("--full", required=True)
    partial.add_argument("--partial", required=True)
    partial.add_argument("--output", required=True)
    partial.add_argument("--metrics", nargs="+", default=["total_reward", "gates_crossed", "full_cycle_success"])

    timing = commands.add_parser("benchmark")
    add_model_arguments(timing)
    timing.add_argument("--output", required=True)
    timing.add_argument("--warmup", type=int, default=100)
    timing.add_argument("--iterations", type=int, default=1000)
    add_environment_arguments(timing)

    plot_metric = commands.add_parser("plot-metric")
    plot_metric.add_argument("result_dirs", nargs="+")
    plot_metric.add_argument("--metric", required=True)
    plot_metric.add_argument("--output", required=True)
    plot_latency = commands.add_parser("plot-latency")
    plot_latency.add_argument("efficiency_jsons", nargs="+")
    plot_latency.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "evaluate":
        checkpoints = args.checkpoint
        if args.labels and len(args.labels) != len(checkpoints):
            raise ValueError("--labels must have one value per checkpoint.")
        labels = args.labels or ([args.label] if len(checkpoints) == 1 else [Path(item).stem for item in checkpoints])
        results = []
        for checkpoint, label in zip(checkpoints, labels, strict=True):
            result_output = Path(args.output) if len(checkpoints) == 1 else Path(args.output) / label
            result = collect_rollouts(RolloutConfig(
                checkpoint=checkpoint, output_dir=str(result_output), architecture=args.architecture,
                label=label, episodes=args.episodes, seed=args.seed, device=args.device,
                record_recurrent=args.record_recurrent, record_internals=args.record_internals,
                env_overrides=environment_overrides(args),
            ))
            plot_failure_breakdown(result, result / "figures" / "failure_breakdown.png")
            results.append(result)
        if len(results) > 1:
            for metric in ("total_reward", "gates_crossed", "full_cycle_success", "policy_action_delta_rms"):
                plot_metric_comparison(results, metric, Path(args.output) / "figures" / f"comparison_{metric}.png")
        print("\n".join(str(result) for result in results))
    elif args.command == "aggregate":
        seed_rows = []
        for directory in args.result_dirs:
            metadata = read_json(Path(directory) / "metadata.json")
            means = flatten_summary_means(read_json(Path(directory) / "summary.json"))
            means.update(
                training_seed=metadata.get("training_seed"), label=metadata.get("experiment_label"),
                within_checkpoint_episode_summary=read_json(Path(directory) / "summary.json"),
            )
            seed_rows.append(means)
        result = aggregate_training_seeds(seed_rows, args.metrics or None)
        write_json(args.output, result)
        print(args.output)
    elif args.command == "compare-partial":
        full_meta = read_json(Path(args.full) / "metadata.json")
        partial_meta = read_json(Path(args.partial) / "metadata.json")
        if full_meta["architecture"] != partial_meta["architecture"]:
            raise ValueError("Partial-observability comparisons must be architecture-matched.")
        full = flatten_summary_means(read_json(Path(args.full) / "summary.json"))
        partial_values = flatten_summary_means(read_json(Path(args.partial) / "summary.json"))
        result = {
            "architecture": full_meta["architecture"],
            "full": args.full,
            "partial": args.partial,
            "comparison": compare_partial_observability(partial_values, full, args.metrics),
        }
        write_json(args.output, result)
        print(args.output)
    elif args.command == "benchmark":
        benchmark_inference(
            args.checkpoint, args.output, architecture=args.architecture, device=args.device,
            seed=args.seed, warmup=args.warmup, iterations=args.iterations,
            env_overrides=environment_overrides(args),
        )
        plot_latency_comparison([Path(args.output) / "efficiency.json"], Path(args.output) / "figures" / "latency.png")
        print(args.output)
    elif args.command == "plot-metric":
        plot_metric_comparison(args.result_dirs, args.metric, args.output)
        print(args.output)
    else:
        plot_latency_comparison(args.efficiency_jsons, args.output)
        print(args.output)


if __name__ == "__main__":
    main()
