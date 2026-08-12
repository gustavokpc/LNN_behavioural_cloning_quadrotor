#!/usr/bin/env python3
"""Export conv_cfc_default_n64_bebop2_delayed_one_step for Paparazzi."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


C_DIR = Path(__file__).resolve().parent
PROJECT = C_DIR.parents[2]
REPO_ROOT = PROJECT.parent
sys.path.insert(0, str(REPO_ROOT))

from LNN_behavioural_cloning_quadrotor.utils.config import load_yaml
from LNN_behavioural_cloning_quadrotor.utils.data import expand_feature_labels
from LNN_behavioural_cloning_quadrotor.utils.normalization_limits import (
    get_normalization_limits,
    resolve_normalization_profile,
)


CHECKPOINT = (
    PROJECT
    / "checkpoints/bebop2/conv_cfc_default_n64_bebop2_delayed_one_step.ckpt"
)
CONFIG = (
    PROJECT
    / "configs/bebop2/conv_cfc_default_n64_bebop2_delayed_one_step.yaml"
)

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


def format_values(values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    lines = []
    for start in range(0, flat.size, 6):
        literals = []
        for value in flat[start : start + 6]:
            literal = f"{float(value):.9g}"
            if "e" not in literal and "E" not in literal and "." not in literal:
                literal += ".0"
            literals.append(literal + "f")
        lines.append("    " + ", ".join(literals) + ",")
    return "\n".join(lines)


def array_definition(name: str, values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    return f"const float {name}[{flat.size}] = {{\n{format_values(flat)}\n}};\n"


def main() -> None:
    config = load_yaml(CONFIG)
    dataset = config["dataset"]
    model = config["model"]
    if "dt" not in dataset["input_labels"]:
        raise ValueError("Expected a checkpoint trained with dt.")
    if model.get("type") != "cfc" or model.get("cfc_mode") != "default":
        raise ValueError("Expected a default-mode CfC.")
    if model.get("activation") != "lecun_tanh":
        raise ValueError("Expected the LeCun-tanh backbone activation.")
    if config.get("sequencing", {}).get("seq_len") != 1:
        raise ValueError("Expected Conv-CfC seq_len=1.")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    checkpoint_config = checkpoint.get("hyper_parameters")
    if checkpoint_config != config:
        raise ValueError("Checkpoint hyperparameters do not exactly match the YAML config.")

    input_order = expand_feature_labels(
        [label for label in dataset["input_labels"] if label not in {"t", "dt"}]
    )
    if len(input_order) != 19:
        raise ValueError(f"Expected 19 physical inputs, got {len(input_order)}.")

    profile = resolve_normalization_profile(dataset.get("bebop_model", "bebop1"))
    norm_min_by_name, norm_max_by_name = get_normalization_limits(profile)
    norm_min = []
    norm_max = []
    for label in input_order:
        if label.startswith("omega"):
            norm_min.append(norm_min_by_name["omega_min"])
            norm_max.append(norm_max_by_name["omega_max"])
        else:
            norm_min.append(norm_min_by_name[label])
            norm_max.append(norm_max_by_name[label])
    norm_min = np.asarray(norm_min, dtype=np.float32)
    norm_max = np.asarray(norm_max, dtype=np.float32)

    header = f"""/*
File: nn_cfc_parameters.h
Generated from: {CHECKPOINT.name}
Model kind: Conv-CfC default, 64 hidden units, LeCun-tanh backbone
Input order: {", ".join(input_order)}
Normalization: {profile}
CfC timespan: runtime dt in seconds (nominal training value: 0.01 s)
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
    for c_name, key in ARRAYS:
        if key not in state_dict:
            raise KeyError(f"Checkpoint is missing tensor {key!r}.")
        header += f"extern const float {c_name}[{state_dict[key].numel()}];\n"
    header += "\n#endif\n"

    source = '#include "nn_cfc_parameters.h"\n\n'
    source += array_definition("input_norm_min", norm_min) + "\n"
    source += array_definition("input_norm_max", norm_max) + "\n"
    for c_name, key in ARRAYS:
        source += array_definition(c_name, state_dict[key].detach().cpu().numpy()) + "\n"

    (C_DIR / "nn_cfc_parameters.h").write_text(header, encoding="utf-8")
    (C_DIR / "nn_cfc_parameters.c").write_text(source, encoding="utf-8")
    print(f"Exported {CHECKPOINT.name} to {C_DIR}")


if __name__ == "__main__":
    main()
