#!/usr/bin/env python3
"""Animate a dataset-command replay with the same OpenCV simulator viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.animation import animate


def animate_replay(input_npz: Path, output: str, draw_path: bool, record: bool, show_waypoint: bool) -> None:
    with np.load(input_npz) as data:
        time = np.asarray(data["time"], dtype=np.float64)
        states_world = np.asarray(data["states_world"], dtype=np.float64)
        actions = np.asarray(data["display_actions"] if "display_actions" in data.files else data["actions"], dtype=np.float64)
        reference_world = np.asarray(data["reference_world"], dtype=np.float64)

    steps = min(len(states_world), len(actions), len(time))
    states_world = states_world[:steps]
    actions = actions[:steps]
    time = time[:steps]
    waypoint = reference_world[-1, 0:3]
    target = np.repeat(waypoint[np.newaxis, :], repeats=steps, axis=0) if show_waypoint else []
    waypoints = np.asarray([waypoint], dtype=np.float64) if show_waypoint else []

    animate(
        t=time,
        x=states_world[:, 0],
        y=states_world[:, 1],
        z=states_world[:, 2],
        phi=states_world[:, 6],
        theta=states_world[:, 7],
        psi=states_world[:, 8],
        u=actions,
        target=target,
        waypoints=waypoints,
        file=output,
        record=record,
        auto_play=record,
        close_on_end=record,
        draw_path=draw_path,
    )


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    default_input = (
        PROJECT_ROOT
        / "organized_plots"
        / "sl_runs"
        / "dataset_command_replay"
        / "hover_dataset_test_bebop2_corrected_traj088_matlab_tau006_replay_collocation_then_hold_5p0s.npz"
    )
    default_output = str(default_input.with_name(f"{default_input.stem}_simulator.mp4"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--record", action="store_true", help="Record MP4 instead of opening the interactive animation.")
    parser.add_argument("--no-record", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--draw-path", action="store_true", default=True)
    parser.add_argument("--no-waypoint", action="store_true", help="Do not draw the dataset final waypoint/target.")
    return parser.parse_args(cli_args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)
    animate_replay(
        input_npz=args.input,
        output=args.output,
        draw_path=args.draw_path,
        record=args.record and not args.no_record,
        show_waypoint=not args.no_waypoint,
    )
    if args.record and not args.no_record:
        print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
