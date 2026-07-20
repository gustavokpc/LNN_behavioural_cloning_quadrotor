#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared controller construction helpers used by train/test/simulators.

All executable entrypoints call into this module so architecture selection is
defined exactly once. That keeps recurrent, liquid, and feedforward baselines
aligned across training, testing, and closed-loop simulation.
"""

from __future__ import annotations

from typing import Sequence

import ncps as kncp
import torch


from .feedforward import FeedForwardSequenceController
from .liquid_networks import CFC, LTC, ConvCfC, MLPCfC
from .standard_networks import CTRNN, GRU, LSTM, SimpleRNN


WIDTH_VARIANTS = {
    "eighth": 1.0 / 8.0,
    "quarter": 1.0 / 4.0,
    "half": 1.0 / 2.0,
    "base": 1.0,
    "double": 2.0,
}


def scaled_int(value: int | None, scale: float, minimum: int = 1) -> int | None:
    """Scale an integer width/depth parameter while respecting a floor."""
    if value is None:
        return None
    return max(minimum, int(round(float(value) * scale)))


def scaled_layers(values: Sequence[int], scale: float, minimum: int = 1) -> list[int]:
    """Scale a full list of hidden-layer sizes with the same multiplier."""
    return [max(minimum, int(round(float(value) * scale))) for value in values]


def resolve_scale_factor(config: dict) -> float:
    """Return the global network scaling factor encoded in the config."""
    model_cfg = config.get("model", {})
    if "scale_factor" in model_cfg:
        return float(model_cfg["scale_factor"])

    model_variant = config.get("model_variant", {})
    variant_name = str(model_variant.get("width", "base")).lower()
    return WIDTH_VARIANTS.get(variant_name, 1.0)


def resolve_model_cfg(config: dict) -> dict:
    """Return model settings, accepting legacy `model_variant.family` configs."""
    model_cfg = dict(config.get("model", {}))
    model_variant = config.get("model_variant", {})

    if "type" not in model_cfg and "family" in model_variant:
        model_cfg["type"] = model_variant["family"]

    if str(model_cfg.get("type", "")).lower() == "cfc":
        if "backbone_units" not in model_cfg and "base_backbone_units" in model_variant:
            model_cfg["backbone_units"] = model_variant["base_backbone_units"]
        if "activation" not in model_cfg and config.get("conv_block", {}).get("value", False):
            model_cfg["activation"] = "relu"
        model_cfg.setdefault("cfc_mode", "default")
        model_cfg.setdefault("mixed_memory", False)

    return model_cfg


def supports_recurrent_state(config: dict) -> bool:
    return str(resolve_model_cfg(config).get("type", "ltc")).lower() not in {"mlp", "ff_mlp", "nn"}


def build_ncp_wiring(model_cfg: dict, output_size: int, scale_factor: float = 1.0):
    """
    Construct an NCP wiring from config.

    The default values match the older hard-coded experiment script so the
    refactored pipeline can reproduce those runs without custom edits.
    """
    ncp_cfg = model_cfg.get("ncp", {})
    inter_neurons = scaled_int(ncp_cfg.get("inter_neurons", 32), scale_factor, minimum=1)
    command_neurons = scaled_int(ncp_cfg.get("command_neurons", 24), scale_factor, minimum=1)
    sensory_fanout = scaled_int(ncp_cfg.get("sensory_fanout", 20), scale_factor, minimum=1)
    inter_fanout = scaled_int(ncp_cfg.get("inter_fanout", 16), scale_factor, minimum=1)
    recurrent_command_synapses = scaled_int(
        ncp_cfg.get("recurrent_command_synapses", 16),
        scale_factor,
        minimum=0,
    )
    motor_fanin = scaled_int(ncp_cfg.get("motor_fanin", 20), scale_factor, minimum=1)

    return kncp.wirings.NCP(
        inter_neurons=inter_neurons,
        command_neurons=command_neurons,
        motor_neurons=output_size,
        sensory_fanout=sensory_fanout,
        inter_fanout=inter_fanout,
        recurrent_command_synapses=recurrent_command_synapses,
        motor_fanin=motor_fanin,
    )


def build_recurrent_module(model_cfg: dict, input_size: int, output_size: int, scale_factor: float = 1.0) -> torch.nn.Module:
    model_type = str(model_cfg.get("type", "ltc")).lower()
    hidden_units = scaled_int(model_cfg.get("no_neurons_layer", 64), scale_factor, minimum=1)

    if model_type == "cfc":
        # CfC exposes a few extra backbone options compared with the other
        # recurrent families, so they are threaded through explicitly here.
        backbone_units = scaled_int(model_cfg.get("backbone_units", hidden_units), scale_factor, minimum=1)
        return CFC(
            input_size=input_size,
            units=hidden_units,
            proj_size=output_size,
            batch_first=True,
            mode=model_cfg.get("cfc_mode", "default"),
            mixed_memory=model_cfg.get("mixed_memory", False),
            activation=model_cfg.get("activation", "lecun_tanh"),
            backbone_units=backbone_units,
            backbone_layers=model_cfg.get("backbone_layers", 1),
            backbone_dropout=model_cfg.get("backbone_dropout", 0.0),
        )

    if model_type == "ltc":
        wiring = kncp.wirings.FullyConnected(hidden_units, output_dim=output_size)
        return LTC(
            input_size=input_size,
            units=wiring,
            batch_first=True,
            ode_unfolds=model_cfg.get("ode_unfolds", 6),
            mixed_memory=model_cfg.get("mixed_memory", False),
        )

    if model_type == "ncp":
        wiring = build_ncp_wiring(model_cfg, output_size, scale_factor=scale_factor)
        return CFC(
            input_size=input_size,
            units=wiring,
            proj_size=output_size,
            batch_first=True,
            mixed_memory=model_cfg.get("mixed_memory", False),
            activation=model_cfg.get("activation", "lecun_tanh"),
        )

    if model_type == "simplernn":
        return SimpleRNN(input_dim=input_size, hidden_dim=hidden_units, output_dim=output_size)

    if model_type == "ctrnn":
        return CTRNN(input_dim=input_size, hidden_dim=hidden_units, output_dim=output_size)

    if model_type == "gru":
        return GRU(input_dim=input_size, hidden_dim=hidden_units, output_dim=output_size)

    if model_type == "lstm":
        return LSTM(input_dim=input_size, hidden_dim=hidden_units, output_dim=output_size)

    if model_type in {"mlp", "ff_mlp", "nn"}:
        hidden_layers = model_cfg.get("hidden_layers")
        if not hidden_layers:
            # Fall back to a simple two-layer MLP sized from the recurrent width
            # knob when no explicit feedforward widths were provided.
            hidden_layers = [hidden_units, hidden_units]
        return FeedForwardSequenceController(
            input_dim=input_size,
            output_dim=output_size,
            hidden_layers=scaled_layers(hidden_layers, scale_factor, minimum=1),
            activation=model_cfg.get("activation", "relu"),
            clamp_output=model_cfg.get("clamp_output", False),
        )

    raise ValueError(f"Unsupported model type '{model_cfg.get('type')}'.")


def build_controller_network(config: dict, input_dim: int, output_dim: int) -> torch.nn.Module:
    """
    Construct the controller architecture encoded by `config`.

    Supported knobs:
    - `model.type`: cfc, ltc, ncp, ctrnn, simplernn, gru, lstm, mlp
    - `model.no_neurons_layer`: recurrent hidden size
    - `model.cfc_mode`: default, pure, no_gate
    - `model.ncp.*`: NCP wiring sizes and fan-in/fan-out
    - `model.backbone_units`, `model.backbone_layers`, `model.backbone_dropout`
    - `model.scale_factor`: global width multiplier for the whole network
    - `conv_block.output_dim` and `mlp_block.no_layers`
    """
    scale_factor = resolve_scale_factor(config)
    mlp_cfg = config.get("mlp_block", {})
    conv_cfg = config.get("conv_block", {})
    model_cfg = resolve_model_cfg(config)
    model_type = str(model_cfg.get("type", "ltc")).lower()

    if model_type in {"mlp", "ff_mlp", "nn"} and (mlp_cfg.get("value") or conv_cfg.get("value")):
        raise ValueError("Feedforward baselines currently do not support conv_block/mlp_block preprocessing.")

    if mlp_cfg.get("value", False):
        mlp_layers = scaled_layers(mlp_cfg.get("no_layers", []), scale_factor, minimum=1)
        if not mlp_layers:
            raise ValueError("MLP block requested but `mlp_block.no_layers` is empty.")
        rnn_input_dim = mlp_layers[-1]
    elif conv_cfg.get("value", False):
        rnn_input_dim = scaled_int(conv_cfg.get("output_dim", conv_cfg.get("base_channels", 256),), scale_factor, minimum=8)
    else:
        rnn_input_dim = input_dim

    recurrent_module = build_recurrent_module(model_cfg, rnn_input_dim, output_dim, scale_factor=scale_factor)

    if model_type in {"mlp", "ff_mlp", "nn"}:
        return recurrent_module

    if mlp_cfg.get("value", False):
        return MLPCfC(
            no_input=input_dim,
            layer_sizes=scaled_layers(mlp_cfg["no_layers"], scale_factor, minimum=1),
            rnn_module=recurrent_module,
        )

    if conv_cfg.get("value", False):
        if not config.get("sequencing", {}).get("value", False):
            raise ValueError("conv_block requires sequencing.value=true.")
        seq_len = int(config["sequencing"]["seq_len"])
        return ConvCfC(
            no_input=seq_len,
            rnn_module=recurrent_module,
            width="base",
            base_channels=scaled_int(conv_cfg.get("output_dim", 256), scale_factor, minimum=8),
        )

    return recurrent_module
