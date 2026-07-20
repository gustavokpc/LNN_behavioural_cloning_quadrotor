#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate the BBP1 Conv-CfC C parameter export from a PyTorch checkpoint."""

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
from LNN_behavioural_cloning_quadrotor.utils.normalization_limits import GLOBAL_MAX, GLOBAL_MIN


ARRAYS = [
    ("conv_block_conv1_weight", "model.conv_block.conv1.weight"),
    ("conv_block_conv1_bias", "model.conv_block.conv1.bias"),
    ("conv_block_conv2_weight", "model.conv_block.conv2.weight"),
    ("conv_block_conv2_bias", "model.conv_block.conv2.bias"),
    ("conv_block_bn2_weight", "model.conv_block.bn2.weight"),
    ("conv_block_bn2_bias", "model.conv_block.bn2.bias"),
    ("conv_block_bn2_running_mean", "model.conv_block.bn2.running_mean"),
    ("conv_block_bn2_running_var", "model.conv_block.bn2.running_var"),
    ("conv_block_conv3_weight", "model.conv_block.conv3.weight"),
    ("conv_block_conv3_bias", "model.conv_block.conv3.bias"),
    ("conv_block_conv4_weight", "model.conv_block.conv4.weight"),
    ("conv_block_conv4_bias", "model.conv_block.conv4.bias"),
    ("conv_block_bn4_weight", "model.conv_block.bn4.weight"),
    ("conv_block_bn4_bias", "model.conv_block.bn4.bias"),
    ("conv_block_bn4_running_mean", "model.conv_block.bn4.running_mean"),
    ("conv_block_bn4_running_var", "model.conv_block.bn4.running_var"),
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
EMBEDDED_CFC_TIMESPAN = 0.01


def _input_order(config: dict) -> list[str]:
    labels = [label for label in config["dataset"]["input_labels"] if label not in {"t", "dt"}]
    return expand_feature_labels(labels)


def _norm_vectors(input_order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mins: list[float] = []
    maxs: list[float] = []
    for label in input_order:
        if label.startswith("omega"):
            mins.append(float(GLOBAL_MIN["omega_min"]))
            maxs.append(float(GLOBAL_MAX["omega_max"]))
        else:
            mins.append(float(GLOBAL_MIN[label]))
            maxs.append(float(GLOBAL_MAX[label]))
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


def write_parameters(checkpoint_path: Path, config_path: Path, output_dir: Path) -> None:
    config = load_yaml(config_path)
    state_dict = _load_state_dict(checkpoint_path)
    input_order = _input_order(config)
    norm_min, norm_max = _norm_vectors(input_order)

    output_dir.mkdir(parents=True, exist_ok=True)
    header_path = output_dir / "nn_parameters.h"
    source_path = output_dir / "nn_parameters.c"

    header = f"""/*
File: nn_parameters.h
Generated from: {checkpoint_path.name}
Model kind: cfc_default conv front-end
Input order: {", ".join(input_order)}
Normalization: LNN_behavioural_cloning_quadrotor.utils.normalization_limits
*/
#ifndef NN_PARAMETERS_H
#define NN_PARAMETERS_H

#define NUM_STATES {len(input_order)}
#define NUM_CONTROLS 4
#define CONV_FEATURES 256
#define HIDDEN_SIZE 64

extern const float input_norm_min[{len(input_order)}];
extern const float input_norm_max[{len(input_order)}];
"""
    for c_name, key in ARRAYS:
        flat_size = int(state_dict[key].detach().cpu().numel())
        header += f"extern const float {c_name}[{flat_size}];\n"
    header += "\n#endif\n"

    source = '#include "nn_parameters.h"\n\n'
    source += _array_definition("input_norm_min", norm_min) + "\n"
    source += _array_definition("input_norm_max", norm_max) + "\n"
    for c_name, key in ARRAYS:
        if key not in state_dict:
            raise KeyError(f"Checkpoint is missing tensor '{key}'.")
        values = state_dict[key].detach().cpu().numpy()
        if c_name in {"rnn_rnn_cell_time_a_weight", "rnn_rnn_cell_time_a_bias"}:
            values = values * EMBEDDED_CFC_TIMESPAN
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
        default=PROJECT / "checkpoints" / "bebop1" / "BBP1_NOISE_conv_cfc_default_n64_bebop1_epoch=19_val_loss=0.000149.ckpt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs" / "bebop1" / "conv_BBP1_NOISE_cfc_default_n64_bebop1_epoch=19_val_loss=0.000149.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=C_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    write_parameters(args.checkpoint, args.config, args.output_dir)


if __name__ == "__main__":
    main()
