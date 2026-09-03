"""Checkpoint discovery/loading via the existing RL construction code."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch


RL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RL_ROOT.parent


def checkpoint_step(path: str | Path) -> int | None:
    match = re.search(r"_(\d+)_steps(?:\.zip)?$", Path(path).name)
    return int(match.group(1)) if match else None


def recover_checkpoint_step(path: str | Path) -> int | None:
    """Recover a real step from filename or exact final-checkpoint metadata."""

    step = checkpoint_step(path)
    if step is not None:
        return step
    metadata = find_training_metadata(path)
    declared_checkpoint = Path(str(metadata.get("checkpoint", ""))).name
    total = metadata.get("total_timesteps")
    if declared_checkpoint == Path(path).name and isinstance(total, (int, float)):
        return int(total)
    return None


def detect_architecture(path: str | Path) -> str:
    """Infer only when directory/file naming is unambiguous."""

    value = str(path).lower()
    ordered = (
        ("recurrent_ppo_ncp_cfc", "recurrent_ppo_ncp_cfc"),
        ("ncp_cfc", "recurrent_ppo_ncp_cfc"),
        ("recurrent_ppo_ltc", "recurrent_ppo_ltc"),
        ("/ltc/", "recurrent_ppo_ltc"),
        ("ct_rnn", "ct_rnn"),
        ("ctrnn", "ct_rnn"),
        ("/rnn/", "rnn"),
        ("/cfc/", "cfc"),
        ("recurrent_ppo", "cfc"),
        ("/ppo/", "ppo"),
        ("/mlp/", "mlp"),
    )
    for marker, architecture in ordered:
        if marker in value:
            return architecture
    raise ValueError("Architecture cannot be inferred reliably; pass --architecture explicitly.")


def find_training_metadata(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    stem = checkpoint.stem
    base_stem = re.sub(r"_\d+_steps$", "", stem)
    matches: list[tuple[int, dict[str, Any]]] = []
    for metadata_path in (RL_ROOT / "runs").glob("**/metadata.json"):
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        candidate = Path(str(data.get("checkpoint", ""))).expanduser()
        experiment_id = str(data.get("experiment_id", ""))
        score = 0
        if candidate.name == checkpoint.name:
            score = 4
        elif candidate.stem == base_stem:
            score = 3
        elif experiment_id and (stem == experiment_id or stem.startswith(experiment_id + "_")):
            score = 2
        if score:
            data["_metadata_path"] = str(metadata_path.resolve())
            matches.append((score, data))
    if not matches:
        return {}
    return max(matches, key=lambda item: item[0])[1]


def default_training_args() -> argparse.Namespace:
    """Obtain current environment defaults from the existing training parser."""

    from ..drone_ppo_bebop2 import parse_args

    with patch.object(sys, "argv", ["analysis"]):
        return parse_args()


def build_args(
    checkpoint: str | Path,
    architecture: str = "auto",
    env_overrides: Mapping[str, Any] | None = None,
    device: str = "auto",
) -> tuple[argparse.Namespace, dict[str, Any]]:
    training_metadata = find_training_metadata(checkpoint)
    args = default_training_args()
    for key, value in training_metadata.items():
        if hasattr(args, key) and value is not None:
            setattr(args, key, value)
    if architecture == "auto":
        architecture = str(training_metadata.get("policy_type") or training_metadata.get("model") or "auto")
        if architecture == "auto":
            architecture = detect_architecture(checkpoint)
    aliases = {"ltc": "recurrent_ppo_ltc", "ncp": "recurrent_ppo_ncp_cfc", "ncp_cfc": "recurrent_ppo_ncp_cfc"}
    args.policy_type = aliases.get(architecture, architecture)
    args.cont = str(Path(checkpoint).resolve())
    args.device = device
    args.num_envs = 1
    args.render = False
    if env_overrides:
        for key, value in env_overrides.items():
            if not hasattr(args, key):
                raise ValueError(f"Unknown environment option: {key}")
            setattr(args, key, value)
    return args, training_metadata


def load_model_and_env(
    checkpoint: str | Path,
    *,
    architecture: str = "auto",
    device: str = "auto",
    seed: int = 0,
    env_overrides: Mapping[str, Any] | None = None,
    policy_dt: float | None = None,
):
    """Load with the same algorithm/policy classes used during training."""

    from ..drone_ppo_bebop2 import (
        _synchronize_recurrent_timespan,
        _torch_load_zipfile_compat,
        make_env,
        resolve_bebop2_algorithm,
    )

    args, metadata = build_args(checkpoint, architecture, env_overrides, device)
    args.seed = int(seed)
    env = make_env(args, num_envs=1, seed=seed)
    algorithm, policy_class, _ = resolve_bebop2_algorithm(args)
    with _torch_load_zipfile_compat():
        model = algorithm.load(
            args.cont,
            env=env,
            device=device,
            custom_objects={"policy_class": policy_class},
        )
    sync_args = argparse.Namespace(**vars(args))
    if policy_dt is not None:
        sync_args.dt = float(policy_dt)
    _synchronize_recurrent_timespan(model, sync_args)
    return model, env, args, metadata


def model_information(model, checkpoint: str | Path) -> dict[str, Any]:
    import torch

    checkpoint = Path(checkpoint)
    packages = {}
    for package in ("torch", "stable-baselines3", "sb3-contrib", "ncps", "numpy", "gymnasium"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    parameters = list(model.policy.parameters())
    return {
        "trainable_parameters": int(sum(parameter.numel() for parameter in parameters if parameter.requires_grad)),
        "total_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "checkpoint_size_bytes": int(checkpoint.stat().st_size),
        "device": str(model.device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "packages": packages,
    }
