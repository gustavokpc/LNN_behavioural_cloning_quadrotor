from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
CHECKPOINT = (
    PROJECT
    / "rl/checkpoints/CFC/cfc_tau_0025/"
    "recurrent_ppo_figure8_gates_cfc_tau_0025_noise_96000000_steps.zip"
)
OPERATIONS_TEMPLATE_DIR = PROJECT / "C_codes/bebop2/RL_CFC_noise_measurements"

TENSORS = {
    "CFC_BACKBONE0_WEIGHT": "lstm_actor.rnn_cell.backbone.0.weight",
    "CFC_BACKBONE0_BIAS": "lstm_actor.rnn_cell.backbone.0.bias",
    "CFC_FF1_WEIGHT": "lstm_actor.rnn_cell.ff1.weight",
    "CFC_FF1_BIAS": "lstm_actor.rnn_cell.ff1.bias",
    "CFC_FF2_WEIGHT": "lstm_actor.rnn_cell.ff2.weight",
    "CFC_FF2_BIAS": "lstm_actor.rnn_cell.ff2.bias",
    "CFC_TIME_A_WEIGHT": "lstm_actor.rnn_cell.time_a.weight",
    "CFC_TIME_A_BIAS": "lstm_actor.rnn_cell.time_a.bias",
    "CFC_TIME_B_WEIGHT": "lstm_actor.rnn_cell.time_b.weight",
    "CFC_TIME_B_BIAS": "lstm_actor.rnn_cell.time_b.bias",
    "POLICY0_WEIGHT": "mlp_extractor.policy_net.0.weight",
    "POLICY0_BIAS": "mlp_extractor.policy_net.0.bias",
    "POLICY2_WEIGHT": "mlp_extractor.policy_net.2.weight",
    "POLICY2_BIAS": "mlp_extractor.policy_net.2.bias",
    "ACTION_WEIGHT": "action_net.weight",
    "ACTION_BIAS": "action_net.bias",
}

EXPECTED_SHAPES = {
    "CFC_BACKBONE0_WEIGHT": (128, 84),
    "CFC_BACKBONE0_BIAS": (128,),
    "CFC_FF1_WEIGHT": (64, 128),
    "CFC_FF1_BIAS": (64,),
    "CFC_FF2_WEIGHT": (64, 128),
    "CFC_FF2_BIAS": (64,),
    "CFC_TIME_A_WEIGHT": (64, 128),
    "CFC_TIME_A_BIAS": (64,),
    "CFC_TIME_B_WEIGHT": (64, 128),
    "CFC_TIME_B_BIAS": (64,),
    "POLICY0_WEIGHT": (64, 64),
    "POLICY0_BIAS": (64,),
    "POLICY2_WEIGHT": (64, 64),
    "POLICY2_BIAS": (64,),
    "ACTION_WEIGHT": (4, 64),
    "ACTION_BIAS": (4,),
}


def load_checkpoint() -> tuple[dict[str, torch.Tensor], dict]:
    with zipfile.ZipFile(CHECKPOINT) as archive:
        state_dict = torch.load(
            io.BytesIO(archive.read("policy.pth")),
            map_location="cpu",
            weights_only=False,
        )
        metadata = json.loads(archive.read("data"))
    return state_dict, metadata


def float_literal(value: np.float32) -> str:
    text = format(float(value), ".9g")
    if "e" not in text and "." not in text:
        text += ".0"
    return text + "f"


def format_array(name: str, values: np.ndarray) -> str:
    flat = values.astype(np.float32, copy=False).reshape(-1)
    lines = []
    for start in range(0, flat.size, 6):
        lines.append("    " + ", ".join(float_literal(v) for v in flat[start : start + 6]))
    return f"const float {name}[{flat.size}] = {{\n" + ",\n".join(lines) + "\n};\n"


def validate_metadata(metadata: dict) -> None:
    if metadata.get("num_timesteps") != 96_000_000:
        raise ValueError(
            f"Expected num_timesteps=96000000, got {metadata.get('num_timesteps')!r}"
        )

    policy_kwargs = metadata.get("policy_kwargs", {})
    if policy_kwargs.get("cfc_timespan") != 0.01:
        raise ValueError(
            f"Expected cfc_timespan=0.01, got {policy_kwargs.get('cfc_timespan')!r}"
        )
    if policy_kwargs.get("lstm_hidden_size") != 64:
        raise ValueError("Checkpoint does not have the expected 64-value recurrent state.")

    observation_space = metadata.get("observation_space", {})
    if observation_space.get("_shape") != [20]:
        raise ValueError("Checkpoint observation shape is not [20].")

    action_space = metadata.get("action_space", {})
    if action_space.get("_shape") != [4]:
        raise ValueError("Checkpoint action shape is not [4].")
    if action_space.get("low") != "[0. 0. 0. 0.]" or action_space.get("high") != "[1. 1. 1. 1.]":
        raise ValueError("Checkpoint action space is not [0, 1].")


def copy_operations() -> None:
    for name in ("rl_cfc_operations.c", "rl_cfc_operations.h"):
        source = OPERATIONS_TEMPLATE_DIR / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing Paparazzi-compatible operations template: {source}")
        shutil.copyfile(source, HERE / name)


def main() -> None:
    state_dict, metadata = load_checkpoint()
    validate_metadata(metadata)

    arrays: dict[str, np.ndarray] = {}
    for c_name, torch_name in TENSORS.items():
        tensor = state_dict[torch_name].detach().cpu().numpy().astype(np.float32)
        if tensor.shape != EXPECTED_SHAPES[c_name]:
            raise ValueError(f"Unexpected shape for {torch_name}: {tensor.shape}")
        arrays[c_name] = tensor

    header_lines = [
        "#ifndef RL_CFC_PARAMETERS_H",
        "#define RL_CFC_PARAMETERS_H",
        "",
        "/* Exact deterministic actor export from:",
        " * recurrent_ppo_figure8_gates_cfc_tau_0025_noise_96000000_steps.zip",
        " * 20 raw observations, 64-value CfC hidden state, four [0, 1] actions,",
        " * and a 100 Hz recurrent timestep (CfC timespan = 0.01 s).",
        " */",
        "#define NUM_STATES 20",
        "#define NUM_CONTROLS 4",
        "#define CFC_INPUT_DIM 20",
        "#define HIDDEN_SIZE 64",
        "#define CFC_BACKBONE_DIM 128",
        "#define POLICY_HIDDEN_DIM 64",
        "#define CFC_TIMESPAN 0.01f",
        "",
    ]
    for name, values in arrays.items():
        header_lines.append(f"extern const float {name}[{values.size}];")
    header_lines.extend(["", "#endif", ""])
    (HERE / "rl_cfc_parameters.h").write_text("\n".join(header_lines))

    source_lines = ['#include "rl_cfc_parameters.h"', ""]
    source_lines.extend(format_array(name, values) for name, values in arrays.items())
    (HERE / "rl_cfc_parameters.c").write_text("\n".join(source_lines))
    copy_operations()

    print(f"Exported exact 96,000,000-step actor to {HERE}")
    print("Paparazzi checkout was not modified.")


if __name__ == "__main__":
    main()
