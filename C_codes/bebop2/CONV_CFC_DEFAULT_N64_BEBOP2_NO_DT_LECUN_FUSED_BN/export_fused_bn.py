#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the no-dt Conv-CfC with BatchNorm folded into conv2 and conv4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


C_DIR = Path(__file__).resolve().parent
PROJECT = C_DIR.parents[2]
REPO_ROOT = PROJECT.parent
sys.path.insert(0, str(REPO_ROOT))

from LNN_behavioural_cloning_quadrotor.utils.config import load_yaml
from LNN_behavioural_cloning_quadrotor.utils.data import expand_feature_labels
from LNN_behavioural_cloning_quadrotor.utils.normalization_limits import get_normalization_limits


ARRAYS = [
    ("conv_block_conv1_weight", "model.conv_block.conv1.weight"),
    ("conv_block_conv1_bias", "model.conv_block.conv1.bias"),
    ("conv_block_conv2_weight", "model.conv_block.conv2.weight"),
    ("conv_block_conv2_bias", "model.conv_block.conv2.bias"),
    ("conv_block_conv3_weight", "model.conv_block.conv3.weight"),
    ("conv_block_conv3_bias", "model.conv_block.conv3.bias"),
    ("conv_block_conv4_weight", "model.conv_block.conv4.weight"),
    ("conv_block_conv4_bias", "model.conv_block.conv4.bias"),
    ("rnn_rnn_cell_backbone_0_weight", "model.rnn.rnn_cell.backbone.0.weight"),
    ("rnn_rnn_cell_backbone_0_bias", "model.rnn.rnn_cell.backbone.0.bias"),
    ("rnn_rnn_cell_ff1_weight", "model.rnn.rnn_cell.ff1.weight"),
    ("rnn_rnn_cell_ff1_bias", "model.rnn.rnn_cell.ff1.bias"),
    ("rnn_rnn_cell_ff2_weight", "model.rnn.rnn_cell.ff2.weight"),
    ("rnn_rnn_cell_ff2_bias", "model.rnn.rnn_cell.ff2.bias"),
    ("rnn_rnn_cell_time_a_weight", "model.rnn.rnn_cell.time_a.weight"),
    ("rnn_rnn_cell_time_a_bias", "model.rnn.rnn_cell.time_a.bias"),
    ("rnn_rnn_cell_time_b_weight", "model.rnn.rnn_cell.time_b.weight"),
    ("rnn_rnn_cell_time_b_bias", "model.rnn.rnn_cell.time_b.bias"),
    ("rnn_fc_weight", "model.rnn.fc.weight"),
    ("rnn_fc_bias", "model.rnn.fc.bias"),
]
BN_EPS = 1.0e-5


def _fold_batch_norm(state_dict: dict[str, torch.Tensor], conv_name: str, bn_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Conv1d weights/biases that include an eval-mode BatchNorm1d."""
    weight = state_dict[f"model.conv_block.{conv_name}.weight"].detach().cpu().float()
    bias = state_dict[f"model.conv_block.{conv_name}.bias"].detach().cpu().float()
    gamma = state_dict[f"model.conv_block.{bn_name}.weight"].detach().cpu().float()
    beta = state_dict[f"model.conv_block.{bn_name}.bias"].detach().cpu().float()
    mean = state_dict[f"model.conv_block.{bn_name}.running_mean"].detach().cpu().float()
    variance = state_dict[f"model.conv_block.{bn_name}.running_var"].detach().cpu().float()
    scale = gamma / torch.sqrt(variance + BN_EPS)
    fused_weight = weight * scale.reshape(-1, 1, 1)
    fused_bias = (bias - mean) * scale + beta
    return fused_weight, fused_bias
def _input_order(config: dict) -> list[str]:
    labels = [label for label in config["dataset"]["input_labels"] if label not in {"t", "dt"}]
    return expand_feature_labels(labels)


def _load_norm_limits(profile: str) -> tuple[dict[str, float], dict[str, float]]:
    return get_normalization_limits(profile)


def _norm_vectors(input_order: list[str], norm_min_by_name: dict[str, float], norm_max_by_name: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    mins: list[float] = []
    maxs: list[float] = []
    for label in input_order:
        if label.startswith("omega"):
            mins.append(float(norm_min_by_name["omega_min"]))
            maxs.append(float(norm_max_by_name["omega_max"]))
        else:
            mins.append(float(norm_min_by_name[label]))
            maxs.append(float(norm_max_by_name[label]))
    return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


def _format_values(values: np.ndarray, indent: str = "    ", per_line: int = 6) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    lines = []
    for start in range(0, flat.size, per_line):
        chunk = flat[start:start + per_line]
        formatted = []
        for value in chunk:
            literal = f"{float(value):.9g}"
            if "e" not in literal and "E" not in literal and "." not in literal:
                literal += ".0"
            formatted.append(f"{literal}f")
        lines.append(indent + ", ".join(formatted) + ",")
    return "\n".join(lines)


def _array_definition(name: str, values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    return f"const float {name}[{flat.size}] = {{\n{_format_values(flat)}\n}};\n"


def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return checkpoint.get("state_dict", checkpoint)


def write_parameters(checkpoint_path: Path, config_path: Path, output_dir: Path, normalization_limits: str) -> None:
    config = load_yaml(config_path)
    if "dt" in config["dataset"]["input_labels"]:
        raise ValueError("This exporter is only for checkpoints trained without dt.")
    model_cfg = config["model"]
    if model_cfg.get("type") != "cfc" or model_cfg.get("cfc_mode") != "default":
        raise ValueError("Expected a default-mode CfC checkpoint.")
    if model_cfg.get("activation") != "lecun_tanh":
        raise ValueError("Expected a lecun_tanh CfC backbone.")
    state_dict = _load_state_dict(checkpoint_path)
    export_values = {c_name: state_dict[key].detach().cpu() for c_name, key in ARRAYS}
    conv2_weight, conv2_bias = _fold_batch_norm(state_dict, "conv2", "bn2")
    conv4_weight, conv4_bias = _fold_batch_norm(state_dict, "conv4", "bn4")
    export_values["conv_block_conv2_weight"] = conv2_weight
    export_values["conv_block_conv2_bias"] = conv2_bias
    export_values["conv_block_conv4_weight"] = conv4_weight
    export_values["conv_block_conv4_bias"] = conv4_bias
    input_order = _input_order(config)
    norm_min_by_name, norm_max_by_name = _load_norm_limits(normalization_limits)
    norm_min, norm_max = _norm_vectors(input_order, norm_min_by_name, norm_max_by_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    header_path = output_dir / "nn_cfc_parameters.h"
    source_path = output_dir / "nn_cfc_parameters.c"

    header = f"""/*
File: nn_cfc_parameters.h
Generated from: {checkpoint_path.name}
Model kind: no-dt Conv-CfC default, 64 hidden units, LeCun backbone
Input order: {", ".join(input_order)}
Normalization: {normalization_limits}
CfC timespan: implicit 1.0 (the checkpoint was trained without dt)
Optimization: BatchNorm2/4 folded into Conv2/4 (epsilon={BN_EPS:g})
*/
#ifndef NN_CFC_PARAMETERS_H
#define NN_CFC_PARAMETERS_H

#define NUM_STATES {len(input_order)}
#define NUM_CONTROLS 4
#define CONV_FEATURES 256
#define HIDDEN_SIZE 64

extern const float input_norm_min[{len(input_order)}];
extern const float input_norm_max[{len(input_order)}];
"""
    for c_name, _ in ARRAYS:
        flat_size = int(export_values[c_name].numel())
        header += f"extern const float {c_name}[{flat_size}];\n"
    header += "\n#endif\n"

    source = '#include "nn_cfc_parameters.h"\n\n'
    source += _array_definition("input_norm_min", norm_min) + "\n"
    source += _array_definition("input_norm_max", norm_max) + "\n"
    for c_name, _ in ARRAYS:
        values = export_values[c_name].numpy()
        source += _array_definition(c_name, values) + "\n"

    header_path.write_text(header, encoding="utf-8")
    source_path.write_text(source, encoding="utf-8")
    print(f"Wrote {header_path}")
    print(f"Wrote {source_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT / "checkpoints" / "bebop2" / "conv_cfc_default_n64_bebop2_no_dt_lecun.ckpt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs" / "bebop2" / "conv_cfc_default_n64_bebop2_no_dt_lecun.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=C_DIR)
    parser.add_argument(
        "--normalization-limits",
        default="bebop2_tau_0_06",
        help="Named normalization profile from utils/normalization_limits.py.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    write_parameters(args.checkpoint, args.config, args.output_dir, args.normalization_limits)


if __name__ == "__main__":
    main()
