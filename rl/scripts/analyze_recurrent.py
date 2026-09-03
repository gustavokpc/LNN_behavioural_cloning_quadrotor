"""Recurrent summaries, event alignment, episode-split probes, and recovery."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..analysis.io import write_json
from ..analysis.linear_probe import fit_episode_ridge_probe
from ..analysis.plotting import plot_event_traces, plot_probe_r2, plot_recovery_trace
from ..analysis.recovery import recovery_metrics
from ..analysis.recurrent_analysis import save_recurrent_analysis
from ..analysis.rollout import load_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summary")
    summary.add_argument("--result", required=True)
    summary.add_argument("--event", default="gate_crossing", choices=(
        "episode_start", "gate_approach", "gate_crossing", "dropout_onset", "observation_restoration", "perturbation_onset", "failure",
    ))
    summary.add_argument("--pre", type=float, default=0.5)
    summary.add_argument("--post", type=float, default=1.0)
    summary.add_argument("--neurons", type=int, nargs="*")
    summary.add_argument("--outcome", choices=("all", "success", "failure"), default="all")

    probe = commands.add_parser("probe")
    probe.add_argument("--result", required=True)
    probe.add_argument("--alpha", type=float, default=1.0)
    probe.add_argument("--test-fraction", type=float, default=0.25)
    probe.add_argument("--seed", type=int, default=0)

    recovery = commands.add_parser("recovery")
    recovery.add_argument("--result", required=True)
    recovery.add_argument("--event", choices=("perturbation_onset", "dropout_onset", "observation_restoration"), default="perturbation_onset")
    recovery.add_argument("--position-band", type=float, default=0.20)
    recovery.add_argument("--rate-band", type=float, default=0.20)
    recovery.add_argument("--dwell", type=float, default=0.25)
    args = parser.parse_args()
    result_dir = Path(args.result)
    if args.command == "summary":
        analysis = save_recurrent_analysis(
            result_dir, event=args.event, pre_s=args.pre, post_s=args.post,
            neuron_indices=args.neurons, outcome=args.outcome,
        )
        event_path = result_dir / f"event_{args.event}.npz"
        if event_path.exists():
            plot_event_traces(event_path, result_dir / "figures" / f"event_{args.event}.png")
        print(result_dir / "recurrent_summary.json")
    elif args.command == "probe":
        steps = load_steps(result_dir)
        if steps["hidden_state"].shape[1] == 0:
            raise ValueError("No hidden state saved; evaluate with --record-recurrent.")
        analysis = fit_episode_ridge_probe(
            steps["hidden_state"], steps["true_state"], steps["episode_id"],
            alpha=args.alpha, test_fraction=args.test_fraction, seed=args.seed,
        )
        write_json(result_dir / "linear_probe.json", analysis)
        plot_probe_r2(analysis, result_dir / "figures" / "linear_probe_r2.png")
        print(result_dir / "linear_probe.json")
    else:
        recovery_metrics(
            result_dir, event=args.event, position_band_m=args.position_band,
            rate_band_rad_s=args.rate_band, dwell_s=args.dwell,
        )
        plot_recovery_trace(result_dir, args.event, result_dir / "figures" / f"recovery_{args.event}.png")
        print(result_dir / f"recovery_{args.event}.json")


if __name__ == "__main__":
    main()
