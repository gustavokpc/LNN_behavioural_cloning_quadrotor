#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure PyTorch benchmark for closed-loop rollouts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from utils.config import load_yaml, resolve_saved_config
from utils.quadrotor_sim import build_lightning_model, generate_starting_conditions, rollout_controller


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Benchmark the PyTorch controller and Python simulator only.")
    parser.add_argument("--config", type=Path, default=default_root / "simulator_config.yaml")
    parser.add_argument("--config-dir", type=Path, default=default_root / "configs")
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    return parser.parse_args(cli_args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)
    config_sim = load_yaml(args.config)
    config_model = load_yaml(resolve_saved_config(config_sim["model_path"], args.config_dir))
    sim_cfg = config_sim["simulation"]
    dt = float(sim_cfg["dt"])
    horizon_steps = args.horizon_steps or int(round(float(sim_cfg["time_simulation"]) / dt))
    seed = int(config_sim.get("seed", 42) if args.seed is None else args.seed)
    starts = generate_starting_conditions(args.runs, seed=seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_lightning_model(config_model, config_sim["model_path"], args.project_root, device)
    use_sequencing = bool(config_model.get("sequencing", {}).get("value", False))
    seq_len = int(config_model.get("sequencing", {}).get("seq_len", 1))

    total_steps = 0
    start_time = time.perf_counter()
    for start_state in starts:
        _, actions = rollout_controller(
            model=model,
            initial_state=start_state,
            input_labels=config_model["dataset"]["input_labels"],
            dt=dt,
            horizon_steps=horizon_steps,
            device=device,
            use_sequencing=use_sequencing,
            seq_len=seq_len,
            integration_method=sim_cfg.get("integration_method", "explicit"),
            implicit_iters=int(sim_cfg.get("implicit_iterations", 5)),
            stop_fn=None,
        )
        total_steps += int(actions.shape[0])

    elapsed = time.perf_counter() - start_time
    print(f"Model: {config_sim['model_path']}")
    print(f"Runs: {args.runs}")
    print(f"Horizon steps: {horizon_steps}")
    print(f"Total simulated steps: {total_steps}")
    print(f"Total time: {elapsed:.6f}s")
    print(f"Average rollout: {elapsed / args.runs:.9f}s")
    print(f"Average step: {elapsed / max(total_steps, 1):.9f}s")


if __name__ == "__main__":
    main()
