#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SB3 policies initialized from the supervised-learning controller."""

from __future__ import annotations

from pathlib import Path

import torch as th
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule

from ..utils.config import load_yaml
from ..utils.data import get_norm_vectors
from ..utils.quadrotor_sim import build_lightning_model
from .legacy_ppo.drone_ppo_sb3 import LogStdClampMixin


class BCActorCriticExtractor(th.nn.Module):
    """Use the supervised controller as PPO actor and a small MLP as critic."""

    def __init__(
        self,
        features_dim: int,
        bc_config_path: str | Path,
        bc_checkpoint_path: str | Path,
        project_root: str | Path,
        value_hidden_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = 4
        self.latent_dim_vf = value_hidden_dim
        self.project_root = Path(project_root)
        self.bc_config = load_yaml(Path(bc_config_path))
        self.bc_model = build_lightning_model(
            self.bc_config,
            str(bc_checkpoint_path),
            self.project_root,
            th.device("cpu"),
        )
        self.bc_model.train()

        input_labels = self.bc_config["dataset"]["input_labels"]
        global_min, global_max = get_norm_vectors(input_labels)
        self.register_buffer("global_min", th.tensor(global_min.reshape(-1), dtype=th.float32))
        self.register_buffer("global_max", th.tensor(global_max.reshape(-1), dtype=th.float32))
        self.value_net = th.nn.Sequential(
            th.nn.Linear(features_dim, value_hidden_dim),
            th.nn.ReLU(),
            th.nn.Linear(value_hidden_dim, value_hidden_dim),
            th.nn.ReLU(),
        )

    def _normalize_obs(self, features: th.Tensor) -> th.Tensor:
        global_min = self.global_min.to(features.device)
        global_max = self.global_max.to(features.device)
        return (features - global_min) / (global_max - global_min + 1e-10)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        obs = self._normalize_obs(features)
        obs = obs.view(obs.shape[0], 1, 1, obs.shape[-1])
        output = self.bc_model(obs)
        action = output[0] if isinstance(output, tuple) else output
        return action.reshape(features.shape[0], -1)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class FrozenBCController(th.nn.Module):
    """Frozen supervised controller used as an action prior for residual PPO."""

    def __init__(
        self,
        bc_config_path: str | Path,
        bc_checkpoint_path: str | Path,
        project_root: str | Path,
        device: str | th.device = "cpu",
    ):
        super().__init__()
        self.device = th.device(device)
        self.project_root = Path(project_root)
        self.bc_config = load_yaml(Path(bc_config_path))
        self.bc_model = build_lightning_model(
            self.bc_config,
            str(bc_checkpoint_path),
            self.project_root,
            self.device,
        )
        self.bc_model.eval()
        for param in self.bc_model.parameters():
            param.requires_grad_(False)

        input_labels = self.bc_config["dataset"]["input_labels"]
        global_min, global_max = get_norm_vectors(input_labels)
        self.register_buffer("global_min", th.tensor(global_min.reshape(-1), dtype=th.float32, device=self.device))
        self.register_buffer("global_max", th.tensor(global_max.reshape(-1), dtype=th.float32, device=self.device))

    def _normalize_obs(self, obs: th.Tensor) -> th.Tensor:
        return (obs - self.global_min) / (self.global_max - self.global_min + 1e-10)

    @th.no_grad()
    def forward(self, obs: th.Tensor) -> th.Tensor:
        obs = obs.to(self.device, dtype=th.float32)
        obs = self._normalize_obs(obs)
        obs = obs.view(obs.shape[0], 1, 1, obs.shape[-1])
        output = self.bc_model(obs)
        action = output[0] if isinstance(output, tuple) else output
        return action.reshape(obs.shape[0], -1).clamp(0.0, 1.0)

    @th.no_grad()
    def predict_numpy(self, obs) -> np.ndarray:
        obs_tensor = th.as_tensor(obs, dtype=th.float32, device=self.device)
        return self.forward(obs_tensor).cpu().numpy()


class BCInitializedActorCriticPolicy(LogStdClampMixin, ActorCriticPolicy):
    """PPO policy whose actor starts from, and trains, a copied SL controller."""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Schedule,
        *args,
        bc_config_path: str | Path,
        bc_checkpoint_path: str | Path,
        project_root: str | Path,
        value_hidden_dim: int = 64,
        max_log_std: float | None = 1.0,
        **kwargs,
    ):
        kwargs.setdefault("features_extractor_class", FlattenExtractor)
        kwargs.setdefault("ortho_init", False)
        kwargs.setdefault("log_std_init", -3.0)
        self.bc_config_path = bc_config_path
        self.bc_checkpoint_path = bc_checkpoint_path
        self.project_root = project_root
        self.value_hidden_dim = value_hidden_dim
        self.max_log_std = max_log_std
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self._clip_log_std_parameter_()
        self._wrap_optimizer_step_with_log_std_cap()

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = BCActorCriticExtractor(
            features_dim=self.features_dim,
            bc_config_path=self.bc_config_path,
            bc_checkpoint_path=self.bc_checkpoint_path,
            project_root=self.project_root,
            value_hidden_dim=self.value_hidden_dim,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = th.nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
