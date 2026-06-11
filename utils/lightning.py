# -*- coding: utf-8 -*-
"""
File: lightning.py

Purpose:
    Defines drone dynamics and a PyTorch Lightning training framework for
    testing various neural networks. Includes continuous-time and
    discrete-time integration of drone state equations (`dydt` and
    `dynamics`), and a LightningModule (`Lightning_Model`) orchestrating
    open-loop and closed-loop training, validation, and testing with
    configurable loss components and logging.

Key Functionality:
    - `dydt`: computes state derivatives based on physical constants
      and control inputs.
    - `dynamics`: integrates state over time steps to simulate position
      sequences.
    - `Lightning_Model`: wraps a neural network model, handling data
      normalization, loss computation (command vs. positional), optimizer
      setup, and storing predictions for analysis.
"""

import torch
import torch.nn as nn
import lightning as L  # PyTorch Lightning core API
import numpy as np
import time
from . import normalization_limits as norm  # physical constants
from .data import get_norm_vectors  # normalization helper

class Lightning_Model(L.LightningModule):
    """
    PyTorch LightningModule for training trajectory flight of drones.

    Manages training, validation, and testing in open-loop vs. closed-loop
    modes and logs losses and predictions for analysis.
    """
    def __init__(self, model, config):
        """Requires specific pytorch model as well as the yaml config file sent as a dictionary"""
        super().__init__()
        # Hyperparameters and model
        self.lr = config['training']['lr']
        self.model = model
        self.criterion = nn.MSELoss()
        self.save_hyperparameters(config)

        # Storage for test-time outputs
        self.all_yhat = []
        self.all_target = []
        self.all_runtime = []

        # Mode determines loss computation
        # Closed-loop training is intentionally retired in this refactor. The
        # controller is trained consistently in open-loop and evaluated in
        # closed-loop inside the simulator scripts.
        self.mode = "open-loop"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        # Check if time features are used
        self.with_time = (('t' in config['dataset']['input_labels'])
                          or ('dt' in config['dataset']['input_labels']))
        # Precompute normalization bounds
        self.global_min, self.global_max = get_norm_vectors(config['dataset']['input_labels'])
        self.global_min, self.global_max = torch.tensor(self.global_min), torch.tensor(self.global_max)
        self.global_min, self.global_max = self.global_min.permute(0, 2, 1), self.global_max.permute(0, 2, 1)

    def forward(self, x, hx=None, timespans=None):
        # Separate time channel if present
        if self.with_time and timespans is None:
            timespans = x[:, :, -1:]
            x = x[:, :, :-1]
        return self.model(x, hx, timespans)

    def configure_optimizers(self):
        # Use Adam optimizer with validation-loss plateau scheduling.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     factor=self.config["training"].get("lr_factor", 0.5),
        #     patience=self.config["training"].get("lr_patience", 3),
        #     min_lr=self.config["training"].get("lr_min", 0.0),
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "monitor": "val_loss",
            #     "interval": "epoch",
            #     "frequency": 1,
            # },
        }

    def training_step(self, batch):
        self.model.train()
        return self._training_step_OL(batch, log_text='train_loss')

    def _split_time_channel(self, x):
        """
        Separate optional time features from the model inputs.

        Training, validation, and testing all consume the same `(x, timespans)`
        representation so there is only one place that needs to understand how
        time is packed into the dataset tensors.
        """
        if not self.with_time:
            return x, None

        if x.ndim == 3:
            return x[:, :, :-1], x[:, :, -1:]

        if x.ndim == 4:
            return x[:, :, :, :-1], x[:, :, :, -1:]

        raise ValueError(f"Unsupported input rank for time splitting: {x.shape}")

    def _unwrap_model_output(self, output):
        """
        Normalize recurrent and feedforward model outputs to a prediction tensor.
        """
        return output[0] if isinstance(output, tuple) else output

    def _forward_open_loop(self, x, timespans=None, stepwise=False):
        """
        Run the controller in the same open-loop pathway used for optimization.

        `stepwise=True` preserves the recurrent one-step rollout used during
        testing and simulation-style benchmarking, while reusing the exact same
        input preparation and output unwrapping as the batch path.
        """
        if not stepwise:
            return self._unwrap_model_output(self.model(x, timespans=timespans))

        hidden_state = None
        predictions = []
        for i in range(x.shape[1]):
            step_timespans = None if timespans is None else timespans[:, i:(i + 1), ...]
            output = self.model(x[:, i:(i + 1), ...], hx=hidden_state, timespans=step_timespans)
            y_hat = self._unwrap_model_output(output)
            hidden_state = output[1] if isinstance(output, tuple) else None
            predictions.append(y_hat)

        return torch.cat(predictions, dim=1)

    def _open_loop_loss(self, y_hat, y):
        """
        Compute the command-space MSE exactly once for every phase.
        """
        x_hat = y_hat.reshape(-1, y.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])
        loss = self.criterion(x_hat, y_flat)
        if torch.isnan(loss):
            loss = torch.tensor(1e3, dtype=loss.dtype, device=loss.device)
        return loss

    def _training_step_OL(self, batch, log_text='train_loss'):
        """Open-loop: optimize motor command MSE only.
        Use log_text as differentiator between training and validation."""
        x, y = batch
        x, timespans = self._split_time_channel(x)
        y_hat = self._forward_open_loop(x, timespans=timespans)
        loss = self._open_loop_loss(y_hat, y)
        self.log(log_text, loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch):
        self.model.eval()
        return self._training_step_OL(batch, log_text='val_loss')

    def test_step(self, batch):
        """
        Run model in inference mode and collect predictions.
        """
        self.model.eval()
        x, y = batch
        x, timespans = self._split_time_channel(x)

        # Test keeps the stepwise rollout because it is closer to the simulator
        # execution path and gives a more representative runtime measurement.
        if x.is_cuda:
            torch.cuda.synchronize()
        start = time.time()
        y_hat = self._forward_open_loop(x, timespans=timespans, stepwise=True)
        if x.is_cuda:
            torch.cuda.synchronize()
        end = time.time()
        self.all_runtime.append((end - start))

        loss = self._open_loop_loss(y_hat, y)
        # Collect all target and prediction points
        self.all_yhat.append(y_hat[0])
        self.all_target.append(y[0])
        self.log('test_loss', loss, sync_dist=True)
        return {'test_loss': loss}

    def backward(self, loss):
        """
        Override backward to retain computation graph if needed.
        """
        if not loss.requires_grad:
            loss.requires_grad=True
        loss.backward(retain_graph=True)
