#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained controller and optionally run automated ablations.

The testing path mirrors training: it reloads the saved config for a checkpoint,
rebuilds the model via the shared builder, evaluates on the requested test set,
and can then rerun the same evaluation under feature-ablation masks.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Iterable

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.ablation import apply_feature_ablation, iter_ablation_specs
from utils.config import load_yaml, resolve_checkpoint, resolve_saved_config
from utils.data import DatasetController, get_data, transform_to_sequence
from utils.lightning import Lightning_Model
from utils.model_builder import build_controller_network


def _prepare_arrays(config_model: dict, config_test: dict) -> tuple[np.ndarray, np.ndarray]:
    if config_test.get("sensitivity_analysis", False):
        starting_trajectory = None
        desired_trajectories = None
    else:
        starting_trajectory = random.randint(0, 999)
        desired_trajectories = config_test["dataset"]["desired_trajectories"]

    # Test-set loading intentionally uses the same label ordering and
    # normalization rules as training so ablations operate on the exact
    # feature tensor the model expects.
    inputs, outputs = get_data(
        input_labels=config_model["dataset"]["input_labels"],
        output_labels=config_model["dataset"]["output_labels"],
        path=config_test["dataset"]["test_path"],
        starting_trajectory=starting_trajectory,
        desired_trajectories=desired_trajectories,
        normalized=config_model["dataset"].get("normalized", True),
        normalization_limits=config_model["dataset"].get(
            "normalization_limits",
            config_model["dataset"].get("bebop_model", "bebop1"),
        ),
    )

    if config_model.get("sequencing", {}).get("value", False):
        seq_len = int(config_model["sequencing"]["seq_len"])
        # Sequence models predict the last step in each extracted history window.
        inputs = transform_to_sequence(inputs, seq_len)
        outputs = outputs[:, :, (seq_len - 1):]

    return inputs, outputs


def _dataloader_from_arrays(inputs: np.ndarray, outputs: np.ndarray, loader_cfg: dict):
    dataset = DatasetController(inputs, outputs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=loader_cfg["batch_size"],
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        drop_last=loader_cfg["drop_last"],
        shuffle=False,
    )


def _build_model(config_model: dict, checkpoint_path: Path) -> Lightning_Model:
    input_dim = int(config_model["dataset"].get("input_dim", config_model["dataset"].get("input_size")))
    output_dim = int(config_model["dataset"].get("output_dim", config_model["dataset"].get("output_size")))
    network = build_controller_network(config_model, input_dim, output_dim)
    model = Lightning_Model(network, config_model)
    # Checkpoints only store the state dict; the full model topology comes from
    # the archived YAML produced during training.
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def evaluate_arrays(config_model: dict,
                    config_test: dict,
                    checkpoint_path: Path,
                    inputs: np.ndarray,
                    outputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = _dataloader_from_arrays(inputs, outputs, config_test["dataloader"])
    model = _build_model(config_model, checkpoint_path)
    # Logging/model-summary are disabled here because this script is typically
    # used for repeated benchmark and ablation sweeps.
    trainer = L.Trainer(logger=False, enable_model_summary=False, devices=1, num_nodes=1)
    start = time.time()
    trainer.test(model, dataloaders=loader)
    duration = time.time() - start
    print(f"Inference wall time: {duration:.2f}s")
    return (
        torch.stack(model.all_yhat).cpu().numpy(),
        torch.stack(model.all_target).cpu().numpy(),
        np.asarray(model.all_runtime),
    )


def _metrics(yhat: np.ndarray, target: np.ndarray, runtime: np.ndarray) -> dict:
    mse_per_sample = np.mean((yhat - target) ** 2, axis=(1, 2))
    return {
        "mse_mean": float(mse_per_sample.mean()),
        "mse_std": float(mse_per_sample.std()),
        "runtime_mean": float(runtime.mean()) if runtime.size else 0.0,
        "runtime_std": float(runtime.std()) if runtime.size else 0.0,
    }


def plot_predictions(yhat: np.ndarray, target: np.ndarray) -> None:
    if yhat.ndim == 3:
        yhat = yhat[0]
    if target.ndim == 3:
        target = target[0]

    fig, axes = plt.subplots(2, 2, figsize=(15, 5))
    axes = axes.flatten()
    for i in range(4):
        axes[i].plot(target[:, i], label="Target")
        axes[i].plot(yhat[:, i], label="Prediction")
        axes[i].set_title(f"u_{i + 1}")
        axes[i].grid(True)
        axes[i].legend()
    fig.tight_layout()
    plt.show()


def run_ablation_suite(config_model: dict,
                       config_test: dict,
                       checkpoint_path: Path,
                       baseline_inputs: np.ndarray,
                       baseline_outputs: np.ndarray) -> list[tuple[str, dict]]:
    ablation_cfg = config_test.get("ablation", {})
    fill_value = float(ablation_cfg.get("fill_value", 0.0))
    results: list[tuple[str, dict]] = []

    for name, features in iter_ablation_specs(ablation_cfg, config_model["dataset"]["input_labels"]):
        # Each ablation reruns the full evaluation with a masked copy of the
        # original test tensor, leaving the baseline arrays untouched.
        ablated_inputs = apply_feature_ablation(
            baseline_inputs,
            config_model["dataset"]["input_labels"],
            features,
            fill_value=fill_value,
        )
        yhat, target, runtime = evaluate_arrays(config_model, config_test, checkpoint_path, ablated_inputs, baseline_outputs)
        results.append((name, _metrics(yhat, target, runtime)))

    return results


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained controller.")
    default_root = Path(__file__).resolve().parent
    parser.add_argument("--test-config", type=Path, default=default_root / "test_config.yaml")
    parser.add_argument("--config-dir", type=Path, default=default_root / "configs")
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--plot", action="store_true", help="Plot the first predicted trajectory.")
    return parser.parse_args(cli_args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)
    config_test = load_yaml(args.test_config)
    config_model = load_yaml(resolve_saved_config(config_test["model_path"], args.config_dir))
    checkpoint_path = resolve_checkpoint(config_test["model_path"], args.project_root)
    print(f"Loading checkpoint from {checkpoint_path}")

    baseline_inputs, baseline_outputs = _prepare_arrays(config_model, config_test)
    yhat, target, runtime = evaluate_arrays(config_model, config_test, checkpoint_path, baseline_inputs, baseline_outputs)
    baseline_metrics = _metrics(yhat, target, runtime)
    print(f"Baseline MSE: {baseline_metrics['mse_mean']:.8f} ± {baseline_metrics['mse_std']:.8f}")
    print(f"Baseline runtime: {baseline_metrics['runtime_mean']:.8f}s ± {baseline_metrics['runtime_std']:.8f}s")

    if config_test.get("ablation", {}).get("enabled", False):
        ablation_results = run_ablation_suite(config_model, config_test, checkpoint_path, baseline_inputs, baseline_outputs)
        for name, metrics in ablation_results:
            print(
                f"Ablation[{name}] MSE={metrics['mse_mean']:.8f} ± {metrics['mse_std']:.8f} | "
                f"runtime={metrics['runtime_mean']:.8f}s ± {metrics['runtime_std']:.8f}s"
            )

    if args.plot or config_test.get("trajectory_plotter", False):
        plot_predictions(yhat, target)


if __name__ == "__main__":
    main()
