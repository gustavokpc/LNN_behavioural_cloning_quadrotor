#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark PyTorch rollouts against the exported C controller rollouts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from utils.c_controller import CController, c_dir_from_model_path
from utils.config import load_yaml, resolve_saved_config
from utils.quadrotor_sim import build_lightning_model, generate_starting_conditions, rollout_controller
from utils.quadrotor_sim_c import rollout_c_controller


def _reached_target(state: np.ndarray, dist_error: float, vel_error: float, ang_error: float) -> bool:
    return (
        np.linalg.norm(state[0:3]) < dist_error
        and np.linalg.norm(state[3:6]) < vel_error
        and np.linalg.norm(state[6:8]) < ang_error
    )


def _benchmark_python(config_sim: dict,
                      config_model: dict,
                      project_root: Path,
                      starts: np.ndarray,
                      horizon_steps: int,
                      device: torch.device) -> tuple[float, int, int]:
    model = build_lightning_model(config_model, config_sim["model_path"], project_root, device)
    sim_cfg = config_sim["simulation"]
    dt = float(sim_cfg["dt"])
    use_sequencing = bool(config_model.get("sequencing", {}).get("value", False))
    seq_len = int(config_model.get("sequencing", {}).get("seq_len", 1))
    failed = 0
    total_steps = 0

    start_time = time.perf_counter()
    for start_state in starts:
        states, actions = rollout_controller(
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
            stop_fn=lambda state, step: _reached_target(
                state,
                sim_cfg["dist_error"],
                sim_cfg["vel_error"],
                sim_cfg["ang_error"],
            ),
        )
        total_steps += int(actions.shape[0])
        if not _reached_target(states[-1], sim_cfg["dist_error"], sim_cfg["vel_error"], sim_cfg["ang_error"]):
            failed += 1

    return time.perf_counter() - start_time, total_steps, failed


def _benchmark_c(config_sim: dict,
                 config_model: dict,
                 project_root: Path,
                 starts: np.ndarray,
                 horizon_steps: int,
                 c_model_dir: Path | None) -> tuple[float, int, int]:
    c_dir = c_model_dir or c_dir_from_model_path(config_sim["model_path"], project_root)
    controller = CController(c_dir)
    sim_cfg = config_sim["simulation"]
    dt = float(sim_cfg["dt"])
    failed = 0
    total_steps = 0

    start_time = time.perf_counter()
    for start_state in starts:
        states, actions = rollout_c_controller(
            controller=controller,
            initial_state=start_state,
            input_labels=config_model["dataset"]["input_labels"],
            dt=dt,
            horizon_steps=horizon_steps,
            integration_method=sim_cfg.get("integration_method", "explicit"),
            implicit_iters=int(sim_cfg.get("implicit_iterations", 5)),
            stop_fn=lambda state, step: _reached_target(
                state,
                sim_cfg["dist_error"],
                sim_cfg["vel_error"],
                sim_cfg["ang_error"],
            ),
        )
        total_steps += int(actions.shape[0])
        if not _reached_target(states[-1], sim_cfg["dist_error"], sim_cfg["vel_error"], sim_cfg["ang_error"]):
            failed += 1

    return time.perf_counter() - start_time, total_steps, failed


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs C-exported controller rollouts.")
    parser.add_argument("--config", type=Path, default=default_root / "simulator_config.yaml")
    parser.add_argument("--config-dir", type=Path, default=default_root / "configs")
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--c-model-dir", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=None,
        help="Override rollout length. Defaults to time_simulation / dt from simulator_config.yaml.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Device for the PyTorch model. CPU is the fairest comparison against the C controller.",
    )
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

    print(f"Model: {config_sim['model_path']}")
    print(f"Runs: {args.runs}")
    print(f"Horizon steps: {horizon_steps}")
    print(f"dt: {dt}")
    print(f"PyTorch device: {device}")
    print()

    python_time, python_steps, python_failed = _benchmark_python(
        config_sim,
        config_model,
        args.project_root,
        starts,
        horizon_steps,
        device,
    )
    c_time, c_steps, c_failed = _benchmark_c(
        config_sim,
        config_model,
        args.project_root,
        starts,
        horizon_steps,
        args.c_model_dir,
    )

    print("PyTorch rollout:")
    print(f"  total time: {python_time:.6f}s")
    print(f"  avg rollout: {python_time / args.runs:.9f}s")
    print(f"  simulated steps: {python_steps}")
    print(f"  avg step: {python_time / max(python_steps, 1):.9f}s")
    print(f"  failed runs: {python_failed}/{args.runs}")
    print()
    print("C rollout:")
    print(f"  total time: {c_time:.6f}s")
    print(f"  avg rollout: {c_time / args.runs:.9f}s")
    print(f"  simulated steps: {c_steps}")
    print(f"  avg step: {c_time / max(c_steps, 1):.9f}s")
    print(f"  failed runs: {c_failed}/{args.runs}")
    print()
    print(f"Speedup total: {python_time / c_time:.3f}x")
    if python_steps == c_steps:
        print(f"Speedup per step: {(python_time / max(python_steps, 1)) / (c_time / max(c_steps, 1)):.3f}x")
    else:
        print("Speedup per step: not reported because the rollouts stopped at different step counts.")


if __name__ == "__main__":
    main()
