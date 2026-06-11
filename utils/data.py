# -*- coding: utf-8 -*-
"""
File: data.py

Purpose:
    This module provides utilities for loading, preprocessing, and transforming hover trajectory
datasets for machine learning workflows. It defines a custom PyTorch Dataset subclass to interface
with DataLoader, functions to normalize feature vectors based on global limits, load and assemble
input/output arrays from .npz files, and convert continuous data into sequential windows suitable
for sequence models.
"""

from torch.utils.data import Dataset
import torch
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .normalization_limits import GLOBAL_MAX, GLOBAL_MIN


def expand_feature_labels(labels):
    """
    Expand composite dataset labels to the actual feature-channel ordering.

    The dataset stores rotor speeds and commands as 4-vectors. Most configs
    refer to them via the shorthand `omega` and `u`; downstream utilities need
    the expanded channel layout to build masks and feature index lookups.
    """
    expanded = []
    for label in labels:
        if label == "omega":
            expanded.extend([f"omega{i}" for i in range(1, 5)])
        elif label == "u":
            expanded.extend([f"u{i}" for i in range(1, 5)])
        else:
            expanded.append(label)
    return expanded


class DatasetController(Dataset):
    """
    PyTorch Dataset wrapper for input/output arrays.

    Attributes:
        input (np.ndarray): Transposed input data of shape (batch_size, in_features, timesteps).
        output (np.ndarray): Transposed output data of shape (batch_size, out_features, timesteps).
        length (int): Number of samples (trajectories).
    """
    def __init__(self, input, output):
        # Transpose to (batch_size, timesteps, features) for PyTorch
        if input.ndim == 3:
            input_array = input.transpose(0, 2, 1)
        elif input.ndim == 4:
            input_array = input.transpose(0, 2, 3, 1)
        else:
            raise ValueError(f"Unsupported input dimensions: {input.shape}")

        if output.ndim == 3:
            output_array = output.transpose(0, 2, 1)
        elif output.ndim == 4:
            output_array = output.transpose(0, 2, 3, 1)
        else:
            raise ValueError(f"Unsupported output dimensions: {output.shape}")

        # Convert once up front so DataLoader workers do not recreate tensors for
        # every single sample fetch.
        self.input = torch.from_numpy(np.ascontiguousarray(input_array)).to(torch.float32)
        self.output = torch.from_numpy(np.ascontiguousarray(output_array)).to(torch.float32)
        self.length = len(input)

    def __len__(self):
        # Return number of trajectories
        return self.length

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]


def get_norm_vectors(labels):
    """
    Build normalization vectors (min and max) for each feature in `labels`.

    Args:
        labels (list[str]): List of feature names to normalize.
    Returns:
        global_min (np.ndarray): Min values reshaped for broadcasting.
        global_max (np.ndarray): Max values reshaped for broadcasting.
    """
    global_min = []
    global_max = []

    # Collect limits for all known keys
    for key in labels:
        if key in GLOBAL_MIN:
            global_min.append(GLOBAL_MIN[key])
            global_max.append(GLOBAL_MAX[key])

    # Extend omega entries if present (4 motors)
    if 'omega' in labels or 'omega1' in labels:
        # Use dedicated min/max for angular velocity
        global_min.extend([GLOBAL_MIN['omega_min']] * 4)
        global_max.extend([GLOBAL_MAX['omega_max']] * 4)

    # Ensure time limits are bounded between 0 and 1 if necessary
    if ('t' in labels) or ('dt' in labels):
        global_min.append(0)
        global_max.append(1)

    # Convert to arrays and reshape to (1, features, 1)
    global_min = np.array(global_min)[None, :, None]
    global_max = np.array(global_max)[None, :, None]

    return global_min, global_max


def get_data(input_labels,
             output_labels,
             path='datasets/hover_dataset.npz',
             starting_trajectory=None,
             desired_trajectories=None,
             normalized=True,
             with_noise=False,
             with_bias=False):
    """
    Load and preprocess specified trajectories from a .npz dataset file.

    Steps:
      1. Load raw arrays for input and output labels.
      2. Compute auxiliary derived features (e.g., distance_error, attitude_error).
      3. Split multi-dimensional signals (e.g., omega, u) into separate channels.
      4. Generate time-step array (t or dt) per trajectory.
      5. Stack features into contiguous arrays and optionally normalize.

    Args:
        input_labels (list[str]): List of input features names.
        output_labels (list[str]): List of output features names.
        path (str): The dataset path where the values can be taken from as .npz format.
        starting_trajectory (int): The trajectory index where to start taking data from.
        desired_trajectories (int): The number of consecutive trajectories to be processed.
        normalized (bool): If normalised is necessary, use this

    Returns:
        reduced_array_input  (np.ndarray): Shape (T_selected, features, timesteps).
        reduced_array_output (np.ndarray): Shape (T_selected, features, timesteps).
    """
    # Determine absolute file path
    cur_path = os.path.dirname(__file__)
    dataset_path = os.path.join(cur_path, '..', path)

    # Load data from .npz file
    print('loading dataset...')
    with np.load(dataset_path) as full_dataset:
        print(len(full_dataset['dx']), 'file trajectories')

        len_dataset = len(full_dataset['dx'])
        dataset_input = {key: full_dataset[key] for key in input_labels}
        dataset_output = {key: full_dataset[key] for key in output_labels}

    # Free memory
    del full_dataset

    if desired_trajectories is None:
        desired_trajectories = len_dataset
    
    if starting_trajectory is None:
        starting_trajectory = 0

    len_traj = dataset_input[input_labels[0]].shape[1]  # number of timesteps per trajectory

    # Compute Euclidean distance error if requested
    if 'distance_error' in input_labels:
        source = dataset_input if 'dx' in dataset_input else dataset_output
        dx, dy, dz = source['dx'], source['dy'], source['dz']
        dataset_input['distance_error'] = np.sqrt(dx**2 + dy**2 + dz**2)

    # Compute attitude error (angular) if requested
    if 'attitude_error' in input_labels:
        source = dataset_input if 'phi' in dataset_input else dataset_output
        phi, theta, psi = source['phi'], source['theta'], source['psi']
        dataset_input['attitude_error'] = np.sqrt(phi**2 + theta**2 + psi**2)

    # Split vector signals into separate channels (rotational speed and motor commands)
    if 'omega' in input_labels:
        omega_array = dataset_input.pop('omega')
        for i in range(4):
            dataset_input[f'omega{i+1}'] = omega_array[:, :, i]

    if 'u' in output_labels:
        u_array = dataset_output.pop('u')
        for i in range(4):
            dataset_output[f'u{i+1}'] = u_array[:, :, i]

    del u_array, omega_array
    # Build time-step or delta-time arrays based on the dataset to be used
    if 't' in input_labels:
        max_time = dataset_input['t'].max(axis=1)
        # If resampled dataset used, a 0.01s dt is constant throughout trajectories
        k = np.floor_divide(max_time, 0.01).astype(int)
        idx = torch.arange(len_traj).unsqueeze(0)
        mask = idx < torch.from_numpy(k).unsqueeze(1)
        dt_array = (mask.float() * 0.01)
        dt_array = dt_array[starting_trajectory:(starting_trajectory + desired_trajectories), :]
        dataset_input.pop('t')
    elif 'dt' in input_labels:
        # Broadcast dt scalar to full trajectory length
        dt_full = np.tile(dataset_input['dt'][:, None], (1, len_traj)) / 2
        dt_array = dt_full[starting_trajectory:(starting_trajectory + desired_trajectories)]
        dataset_input.pop('dt')

    # External moments: broadcast to time dimension
    for m in ['Mx_ext', 'My_ext', 'Mz_ext']:
        if m in input_labels:
            dataset_input[m] = np.tile(dataset_input[m][:, None], (1, len_traj))
        if m in output_labels:
            dataset_output[m] = np.tile(dataset_output[m][:, None], (1, len_traj))

    # Stack all features into arrays: (samples, features, timesteps)
    reduced_array_input = np.stack(list(dataset_input.values()), axis=1)
    del dataset_input

    reduced_array_output = np.stack(list(dataset_output.values()), axis=1)
    del dataset_output

    # Select subset of trajectories for training/analysis
    reduced_array_input = reduced_array_input[starting_trajectory:(starting_trajectory + desired_trajectories)]
    reduced_array_output = reduced_array_output[starting_trajectory:(starting_trajectory + desired_trajectories)]

    # Append dt channel back if used
    if 't' in input_labels or 'dt' in input_labels:
        dt_channel = np.expand_dims(dt_array, axis=1)
        reduced_array_input = np.concatenate([reduced_array_input, dt_channel], axis=1)

    # Normalize features to [0,1] range if requested
    if normalized:
        global_min, global_max = get_norm_vectors(input_labels)
        reduced_array_input = (reduced_array_input - global_min) / (global_max - global_min + 1e-10)

    if with_noise:
        # Add Gaussian noise to input features (except angular velocities)
        noisy_keys = ['dx', 'dy', 'dz', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'p', 'q', 'r']
        noise_std = 0.001  # Standard deviation of the noise
        noise = np.random.normal(0, noise_std, reduced_array_input.shape)
        for i, key in enumerate(dataset_input.keys()):
            if key in noisy_keys:
                dist_type = np.random.choice(["normal", "laplace", "uniform", "poisson"])
                scale = np.random.uniform(0.0005, 0.003)

                if dist_type == "normal":
                    noise = np.random.normal(0, scale, reduced_array_input[:, i:(i+1), :].shape)
                elif dist_type == "laplace":
                    noise = np.random.laplace(0, scale, reduced_array_input[:, i:(i+1), :].shape)
                elif dist_type == "poisson":
                    noise = np.random.poisson(scale, reduced_array_input[:, i:(i+1), :].shape)
                else:  # uniform
                    noise = np.random.uniform(-scale, scale, reduced_array_input[:, i:(i+1), :].shape)

                reduced_array_input[:, i:(i+1), :] += noise
        reduced_array_input += noise

    if with_bias:
        # Add constant bias to input features (except angular velocities)
        bias_value = 0.005  # Bias value to be added
        bias = np.random.normal(0, bias_value, reduced_array_input.shape[0:2])  # Same bias across time
        fraction = 0.9  # Fraction of values to be set to zero
        n_total = bias.size
        n_zero = int(fraction * n_total)

        # Randomly choose flat indices
        idx = np.random.choice(n_total, n_zero, replace=False)
        # Convert to 2D indices
        rows, cols = np.unravel_index(idx, bias.shape)

        # Set those entries to 0
        bias[rows, cols] = 0.0
        bias = np.expand_dims(bias, axis=2)  # Expand to (samples, features, 1)
        for i, key in enumerate(dataset_input.keys()): # No bias for angular velocities
            if 'M' in key:
                bias[:, i, :] = 0  # No bias for angular velocities
        reduced_array_input += bias

    return reduced_array_input, reduced_array_output


def transform_to_sequence(data, seq_len=100):
    """
    Convert continuous time-series data into overlapping sequences.

    Args:
        data (np.ndarray): Array shaped (samples, features, timesteps).
        seq_len (int): Desired window length for each sequence.
    Returns:
        data_seq (np.ndarray): Flattened array of shape
            (samples * (timesteps - seq_len), seq_len, features).
    """
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1.")

    if data.ndim != 3:
        raise ValueError(f"Expected input data with shape (samples, features, timesteps), got {data.shape}.")

    _, _, sim_len = data.shape
    if seq_len > sim_len:
        raise ValueError(f"seq_len={seq_len} is longer than available trajectory length {sim_len}.")

    # Vectorised sliding window extraction over the timestep axis. Resulting
    # layout is (samples, features, window_count, seq_len), matching the legacy
    # convention used by the convolutional controllers.
    return sliding_window_view(data, window_shape=seq_len, axis=2)
