#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a controller from a single YAML configuration file.

This is the main training entrypoint for the refactored pipeline. The script:
1. loads the train/validation arrays described in `train_config.yaml`,
2. applies sequence slicing and optional delayed-input preprocessing,
3. reconstructs the requested network architecture from the config,
4. trains via the shared Lightning wrapper, and
5. exports both the best checkpoint and its exact resolved config.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Iterable

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from utils.ablation import DEFAULT_FEATURE_GROUPS, drop_features_from_labels, resolve_ablation_features
from utils.config import ensure_dir, load_yaml, save_yaml
from utils.data import DatasetController, get_data, transform_to_sequence
from utils.lightning import Lightning_Model
from utils.model_builder import build_controller_network, resolve_scale_factor


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_sequence(data: np.ndarray, seq_len: int) -> np.ndarray:
    return transform_to_sequence(data, seq_len) if seq_len > 0 else data


def _sanitize_name_part(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()


def _default_checkpoint_name(config: dict) -> str:
    parts: list[str] = []
    if config.get("mlp_block", {}).get("value", False):
        parts.append("mlpblock")
    elif config.get("conv_block", {}).get("value", False):
        parts.append("conv")

    model_cfg = config.get("model", {})
    model_type = str(model_cfg.get("type", "controller")).lower()
    parts.append(model_type)

    if model_type == "cfc":
        parts.append(str(model_cfg.get("cfc_mode", "default")))
    elif model_type == "ncp":
        ncp_cfg = model_cfg.get("ncp", {})
        inter_neurons = ncp_cfg.get("inter_neurons")
        command_neurons = ncp_cfg.get("command_neurons")
        if inter_neurons is not None and command_neurons is not None:
            parts.append(f"ncp{inter_neurons + command_neurons}")

    neurons = model_cfg.get("no_neurons_layer")
    if neurons is not None and model_type not in {"mlp", "ff_mlp", "nn", "ncp"}:
        parts.append(f"n{neurons}")

    scale_factor = resolve_scale_factor(config)
    if abs(scale_factor - 1.0) > 1e-9:
        parts.append(f"scale{scale_factor:g}")

    train_ablation_cfg = config.get("train_ablation", {})
    for tag in train_ablation_cfg.get("name_tags", []):
        parts.append(str(tag))

    dataset_cfg = config.get("dataset", {})
    norm_profile = dataset_cfg.get("normalization_limits", dataset_cfg.get("bebop_model", "bebop1"))
    parts.append(str(norm_profile).strip())

    return "_".join(_sanitize_name_part(part) for part in parts if str(part).strip()) or "controller"


def _prepare_training_ablation(config: dict) -> dict:
    """
    Apply config-driven input-feature removal before any data is loaded.

    This is the training-time ablation path: the model is trained on a reduced
    input space, the reduced label list is persisted into the saved config, and
    the checkpoint name gets a `no_<group>` suffix.
    """
    ablation_cfg = config.setdefault("train_ablation", {})
    if not ablation_cfg.get("enabled", False):
        ablation_cfg["applied_groups"] = []
        ablation_cfg["applied_features"] = []
        ablation_cfg["removed_expanded_features"] = []
        ablation_cfg["name_tags"] = []
        return config

    requested_groups = list(ablation_cfg.get("remove_feature_groups", []))
    requested_features = list(ablation_cfg.get("remove_features", []))
    resolved_features = resolve_ablation_features(
        requested_groups + requested_features,
        feature_sets=ablation_cfg.get("feature_sets", DEFAULT_FEATURE_GROUPS),
    )

    filtered_labels, removed_expanded = drop_features_from_labels(
        config["dataset"]["input_labels"],
        resolved_features,
    )
    if not removed_expanded:
        raise ValueError("train_ablation was enabled, but no matching dataset input features were removed.")

    config["dataset"]["input_labels"] = filtered_labels
    ablation_cfg["applied_groups"] = requested_groups
    ablation_cfg["applied_features"] = resolved_features
    ablation_cfg["removed_expanded_features"] = removed_expanded

    name_tags = [f"no_{group}" for group in requested_groups]
    explicit_only = [
        feature for feature in resolved_features
        if feature not in resolve_ablation_features(requested_groups, ablation_cfg.get("feature_sets", DEFAULT_FEATURE_GROUPS))
    ]
    if explicit_only:
        name_tags.append("no_" + "-".join(_sanitize_name_part(feature) for feature in explicit_only))
    ablation_cfg["name_tags"] = name_tags
    return config


def _prepare_arrays(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_cfg = config["dataset"]
    # Training and validation use the same preprocessing path so saved configs
    # remain directly reusable at test and simulator time.
    train_input, train_output = get_data(
        input_labels=dataset_cfg["input_labels"],
        output_labels=dataset_cfg["output_labels"],
        path=dataset_cfg["train_path"],
        normalized=dataset_cfg.get("normalized", True),
        with_noise=dataset_cfg.get("with_noise", False),
        with_bias=dataset_cfg.get("with_bias", False),
        normalization_limits=dataset_cfg.get(
            "normalization_limits", dataset_cfg.get("bebop_model", "bebop1")
        ),
    )
    val_input, val_output = get_data(
        input_labels=dataset_cfg["input_labels"],
        output_labels=dataset_cfg["output_labels"],
        path=dataset_cfg["val_path"],
        normalized=dataset_cfg.get("normalized", True),
        normalization_limits=dataset_cfg.get(
            "normalization_limits", dataset_cfg.get("bebop_model", "bebop1")
        ),
    )

    if config.get("sequencing", {}).get("value", False):
        seq_len = int(config["sequencing"]["seq_len"])
        # Convolutional and history-based models consume sliding windows rather
        # than raw trajectories, so labels are shifted to the final step.
        train_input = _maybe_sequence(train_input, seq_len)
        val_input = _maybe_sequence(val_input, seq_len)
        train_output = train_output[:, :, (seq_len - 1):]
        val_output = val_output[:, :, (seq_len - 1):]

    if dataset_cfg.get("with_delay", False):
        delay = int(dataset_cfg.get("delay_steps", 2))
        # Artificial sensor delay is implemented by trimming the newest inputs
        # and aligning the targets `delay` steps later in time.
        if train_input.ndim == 4:
            train_input = train_input[:, :, :-delay, :]
            val_input = val_input[:, :, :-delay, :]
        else:
            train_input = train_input[:, :, :-delay]
            val_input = val_input[:, :, :-delay]
        train_output = train_output[:, :, delay:]
        val_output = val_output[:, :, delay:]

    return train_input, train_output, val_input, val_output


def _build_dataloaders(config: dict) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, int]:
    train_input, train_output, val_input, val_output = _prepare_arrays(config)
    train_set = DatasetController(train_input, train_output)
    val_set = DatasetController(val_input, val_output)

    # Stored dataset dimensions exclude the explicit time channel because the
    # recurrent modules receive `dt/t` separately through `timespans`.
    input_dim = int(train_set.input.shape[-1])
    if any(label in {"t", "dt"} for label in config["dataset"]["input_labels"]):
        input_dim -= 1
    output_dim = int(train_set.output.shape[-1])

    loader_cfg = config["dataloader"]
    loader_kwargs = {
        "batch_size": loader_cfg["batch_size"],
        "num_workers": loader_cfg["num_workers"],
        "pin_memory": loader_cfg["pin_memory"],
        "drop_last": loader_cfg["drop_last"],
    }
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, input_dim, output_dim


def train_controller(config_path: Path, project_root: Path) -> Path:
    config = load_yaml(config_path)
    config = _prepare_training_ablation(config)
    _set_seed(int(config.get("training", {}).get("seed", 42)))

    if config.get("train_ablation", {}).get("enabled", False):
        removed = ", ".join(config["train_ablation"]["removed_expanded_features"])
        print(f"Training-time ablation enabled. Removed input features: {removed}")

    train_loader, val_loader, input_dim, output_dim = _build_dataloaders(config)
    config["dataset"]["input_dim"] = input_dim
    config["dataset"]["input_size"] = input_dim
    config["dataset"]["output_dim"] = output_dim
    config["dataset"]["output_size"] = output_dim

    # Model construction is centralized so train/test/simulators all rebuild
    # exactly the same architecture from the saved YAML.
    network = build_controller_network(config, input_dim, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Lightning_Model(network, config).to(device)

    checkpoint_dir = ensure_dir(project_root / "checkpoints")
    checkpoint_name = config.get("checkpoint_name") or _default_checkpoint_name(config)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(checkpoint_dir),
        filename=checkpoint_name + "_{epoch:02d}_{val_loss:.6f}",
        save_top_k=1,
        mode="min",
    )

    logger = None
    if config["training"].get("logger", False):
        logger = WandbLogger(
            project=config["training"].get("wandb_project", "Autonomous-Quadrotor-Training"),
            name=config.get("experiment_name"),
            log_model=True,
        )

    trainer = L.Trainer(
        max_epochs=int(config["training"]["max_epochs"]),
        callbacks=[checkpoint_callback],
        logger=logger,
        devices=1,
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = Path(checkpoint_callback.best_model_path)
    if not best_ckpt.exists():
        raise RuntimeError("Training finished without producing a checkpoint.")

    configs_dir = ensure_dir(project_root / "configs")
    # Export the resolved config next to the checkpoint stem so evaluation only
    # needs a single model name.
    save_yaml(configs_dir / f"{best_ckpt.stem}.yaml", config)
    print(f"Best checkpoint saved at: {best_ckpt}")
    print(f"Configuration saved to: {configs_dir / f'{best_ckpt.stem}.yaml'}")
    return best_ckpt


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train controller from YAML config.")
    default_root = Path(__file__).resolve().parent
    parser.add_argument("--config", type=Path, default=default_root / "train_config.yaml")
    parser.add_argument("--project-root", type=Path, default=default_root)
    return parser.parse_args(cli_args)


def main(cli_args: Iterable[str] | None = None) -> Path:
    args = parse_args(cli_args)
    return train_controller(args.config, args.project_root)


if __name__ == "__main__":
    main()
