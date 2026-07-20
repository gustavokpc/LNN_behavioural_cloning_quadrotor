#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared filesystem and YAML helpers for the training package.

These helpers keep path resolution logic out of the entry scripts so training,
testing, and simulators all resolve checkpoints and saved configs the same way.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def ensure_dir(path: Path) -> Path:
    """Create a directory tree if needed and return the same path object."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict:
    """Read a YAML file using UTF-8 and return the parsed dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, payload: dict) -> None:
    """Write a YAML payload, creating parent directories on demand."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def resolve_checkpoint(model_name: str, project_root: Path) -> Path:
    """
    Resolve a checkpoint path from an experiment stem or explicit path.

    The helper accepts:
    - an absolute or relative file path
    - a stem with or without the `.ckpt` suffix
    - a file living in `<project_root>/checkpoints`
    """
    candidate = Path(model_name)
    if candidate.is_file():
        return candidate

    variants = [candidate]
    if candidate.suffix != ".ckpt":
        variants.append(candidate.with_suffix(".ckpt"))

    checkpoint_dir = project_root / "checkpoints"
    for variant in variants:
        direct = checkpoint_dir / variant.name
        if direct.is_file():
            return direct
        matches = sorted(checkpoint_dir.glob(f"*/{variant.name}"))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Unable to locate checkpoint '{model_name}'.")


def resolve_saved_config(model_name: str, config_dir: Path) -> Path:
    """Resolve a saved training config by experiment stem or explicit path."""
    candidate = Path(model_name)
    if candidate.is_file():
        return candidate

    variants = [candidate]
    if candidate.suffix != ".yaml":
        variants.append(candidate.with_suffix(".yaml"))

    for variant in variants:
        direct = config_dir / variant.name
        if direct.is_file():
            return direct
        matches = sorted(config_dir.glob(f"*/{variant.name}"))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Unable to locate saved config for '{model_name}'.")


def dataset_dims(config: dict) -> tuple[int, int]:
    """Read persisted dataset dimensions, accepting both old and new key names."""
    dataset_cfg = config.get("dataset", {})
    input_dim = dataset_cfg.get("input_dim", dataset_cfg.get("input_size"))
    output_dim = dataset_cfg.get("output_dim", dataset_cfg.get("output_size"))
    if input_dim is None or output_dim is None:
        raise KeyError("Model config is missing dataset input/output dimensions.")
    return int(input_dim), int(output_dim)
