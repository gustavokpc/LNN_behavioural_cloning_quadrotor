#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure-8 gate environment using the Bebop2 dynamics.

The task layout mirrors ``legacy_ppo.Quadcopter3DGates``:

- figure-8 gate positions and gate yaw values
- observations in the target-gate frame
- configurable policy action range, converted to Bebop2 ``[0, 1]`` motor commands
- gate-plane pass/collision reward logic

Internally, the environment keeps a 19-value Bebop2 simulator state and advances
it through ``quadrotor_sim_matlab``.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from ...utils.dynamics_models import quadrotor_sim_matlab, set_dynamics_model
from ...utils.quadrotor_sim import body_to_world_state, integrate_state, world_to_body_state


FIGURE8_RADIUS = 1.5
DEFAULT_FIGURE8_GATE_POS = np.array(
    [
        [FIGURE8_RADIUS, -FIGURE8_RADIUS, -1.5],
        [0.0, 0.0, -1.5],
        [-FIGURE8_RADIUS, FIGURE8_RADIUS, -1.5],
        [0.0, 2.0 * FIGURE8_RADIUS, -1.5],
        [FIGURE8_RADIUS, FIGURE8_RADIUS, -1.5],
        [0.0, 0.0, -1.5],
        [-FIGURE8_RADIUS, -FIGURE8_RADIUS, -1.5],
        [0.0, -2.0 * FIGURE8_RADIUS, -1.5],
    ],
    dtype=np.float32,
)
DEFAULT_FIGURE8_GATE_YAW = np.array([1, 2, 1, 0, -1, -2, -1, 0], dtype=np.float32) * np.pi / 2.0
DEFAULT_FIGURE8_START_POS = DEFAULT_FIGURE8_GATE_POS[0].copy()


class Bebop2Figure8GatesEnv(VecEnv):
    """SB3 VecEnv with figure-8 gate observations and Bebop2 physics."""

    def __init__(
        self,
        num_envs: int,
        gates_pos: np.ndarray | Sequence[Sequence[float]] = DEFAULT_FIGURE8_GATE_POS,
        gate_yaw: np.ndarray | Sequence[float] = DEFAULT_FIGURE8_GATE_YAW,
        start_pos: np.ndarray | Sequence[float] = DEFAULT_FIGURE8_START_POS,
        start_pos_jitter: float = 0.5,
        start_gate: int = 0,
        gates_ahead: int = 1,
        gate_size: float = 1.5,
        motor_limit: float = 1.0,
        initialize_at_random_gates: bool = False,
        initialize_uniform: bool = False,
        num_state_history: int = 0,
        num_action_history: int = 0,
        history_step_size: int = 1,
        param_input: bool = False,
        param_input_noise: float = 0.0,
        low_obs: bool = False,
        no_vel: bool = False,
        no_ang_vel: bool = False,
        dt: float = 0.01,
        max_steps: int = 1200,
        integration_method: str = "rk4",
        implicit_iters: int = 1,
        randomize_external_moments: bool = False,
        action_range: str = "0_1",
        tau: float = 0.06,
        seed: int | None = None,
    ):
        self.seed(seed)
        set_dynamics_model("quadrotor_sim_matlab")

        quadrotor_sim_matlab.TAU = tau

        self.gate_pos = np.asarray(gates_pos, dtype=np.float32)
        self.gate_yaw = np.asarray(gate_yaw, dtype=np.float32)
        self.start_pos = np.asarray(start_pos, dtype=np.float32)
        self.start_pos_jitter = float(start_pos_jitter)
        if self.start_pos_jitter < 0.0:
            raise ValueError("start_pos_jitter must be non-negative.")
        self.num_gates = int(self.gate_pos.shape[0])
        self.start_gate = int(start_gate)
        if not 0 <= self.start_gate < self.num_gates:
            raise ValueError(f"start_gate must be between 0 and {self.num_gates - 1}.")
        self.gates_ahead = int(gates_ahead)
        self.gate_size = float(gate_size)
        self.motor_limit = float(motor_limit)
        self.initialize_at_random_gates = bool(initialize_at_random_gates)
        self.initialize_uniform = bool(initialize_uniform)
        self.num_state_history = int(num_state_history)
        self.num_action_history = int(num_action_history)
        self.history_step_size = int(history_step_size)
        self.param_input = bool(param_input)
        self.param_input_noise = float(param_input_noise)
        self.low_obs = bool(low_obs)
        self.no_vel = bool(no_vel)
        self.no_ang_vel = bool(no_ang_vel)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.integration_method = integration_method
        self.implicit_iters = int(implicit_iters)
        self.randomize_external_moments = bool(randomize_external_moments)
        if action_range not in {"0_1", "neg1_1"}:
            raise ValueError("action_range must be '0_1' or 'neg1_1'.")
        self.action_range = action_range

        self.gate_pos_rel = np.zeros((self.num_gates, 3), dtype=np.float32)
        self.gate_yaw_rel = np.zeros(self.num_gates, dtype=np.float32)
        for idx in range(self.num_gates):
            self.gate_pos_rel[idx] = self.gate_pos[idx] - self.gate_pos[idx - 1]
            rotation = np.array(
                [
                    [np.cos(self.gate_yaw[idx - 1]), np.sin(self.gate_yaw[idx - 1])],
                    [-np.sin(self.gate_yaw[idx - 1]), np.cos(self.gate_yaw[idx - 1])],
                ],
                dtype=np.float32,
            )
            self.gate_pos_rel[idx, 0:2] = rotation @ self.gate_pos_rel[idx, 0:2]
            yaw_rel = float(self.gate_yaw[idx] - self.gate_yaw[idx - 1])
            yaw_rel %= 2.0 * np.pi
            if yaw_rel > np.pi:
                yaw_rel -= 2.0 * np.pi
            elif yaw_rel < -np.pi:
                yaw_rel += 2.0 * np.pi
            self.gate_yaw_rel[idx] = yaw_rel

        if self.action_range == "neg1_1":
            action_low = -1.0
            action_high = 2.0 * self.motor_limit - 1.0
        else:
            action_low = 0.0
            action_high = self.motor_limit
        action_space = spaces.Box(low=action_low, high=action_high, shape=(4,), dtype=np.float32)

        self.state_len = 16 + 4 * self.gates_ahead + 4 * self.num_action_history + 9 * int(self.param_input)
        self.low_obs_state_len = self.state_len - 6
        self.no_vel_state_len = self.state_len - 3
        self.no_ang_vel_state_len = self.state_len - 3
        if self.low_obs:
            self.obs_len = self.low_obs_state_len * (1 + self.num_state_history)
        elif self.no_vel:
            self.obs_len = self.no_vel_state_len * (1 + self.num_state_history)
        elif self.no_ang_vel:
            self.obs_len = self.no_ang_vel_state_len * (1 + self.num_state_history)
        else:
            self.obs_len = self.state_len * (1 + self.num_state_history)
        observation_space = spaces.Box(
            low=np.asarray([-np.inf] * self.obs_len, dtype=np.float32),
            high=np.asarray([np.inf] * self.obs_len, dtype=np.float32),
            shape=(self.obs_len,),
            dtype=np.float32,
        )
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        self.sim_states = np.zeros((num_envs, 19), dtype=np.float32)
        self.world_states = np.zeros((num_envs, 16), dtype=np.float32)
        self.states = np.zeros((num_envs, self.obs_len), dtype=np.float32)
        hist_len = max(40, (self.num_state_history + 1) * max(1, self.history_step_size) + 1)
        self.state_hist = np.zeros((num_envs, hist_len, self.state_len), dtype=np.float32)
        self.action_hist = np.zeros((num_envs, hist_len, 4), dtype=np.float32)
        self.actions = np.zeros((num_envs, 4), dtype=np.float32)
        self.prev_actions = np.zeros((num_envs, 4), dtype=np.float32)
        self.prev_derivs: list[np.ndarray | None] = [None] * num_envs
        self.target_gates = np.zeros(num_envs, dtype=np.int64)
        self.step_counts = np.zeros(num_envs, dtype=np.int64)
        self.dones = np.zeros(num_envs, dtype=bool)

        self.episode_distance = np.zeros(num_envs, dtype=np.float64)
        self.episode_speed_sum = np.zeros(num_envs, dtype=np.float64)
        self.episode_speed_max = np.zeros(num_envs, dtype=np.float64)
        self.episode_bank_angle_sum = np.zeros(num_envs, dtype=np.float64)
        self.episode_action_sum = np.zeros(num_envs, dtype=np.float64)
        self.episode_action_sq_sum = np.zeros(num_envs, dtype=np.float64)
        self.episode_action_count = np.zeros(num_envs, dtype=np.int64)
        self.episode_action_rpm_diff_sum = np.zeros(num_envs, dtype=np.float64)
        self.episode_metric_steps = np.zeros(num_envs, dtype=np.int64)

        self.param_encoding = np.zeros((num_envs, 9), dtype=np.float32)
        self.reset()

    def _omega_to_legacy_motor(self, omega: np.ndarray) -> np.ndarray:
        info = quadrotor_sim_matlab.INFO
        return 2.0 * (omega - info.omega_min) / (info.omega_max - info.omega_min) - 1.0

    def _legacy_motor_to_omega(self, motor_state: np.ndarray) -> np.ndarray:
        info = quadrotor_sim_matlab.INFO
        return info.omega_min + 0.5 * (motor_state + 1.0) * (info.omega_max - info.omega_min)

    def action_to_bebop_command(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions)
        if self.action_range == "neg1_1":
            actions = (actions + 1.0) * 0.5
        return np.clip(actions, 0.0, self.motor_limit)

    def _sample_external_moments(self, num_samples: int) -> np.ndarray:
        if not self.randomize_external_moments:
            return np.zeros((num_samples, 3), dtype=np.float64)
        return np.random.uniform(-0.01, 0.01, size=(num_samples, 3)).astype(np.float64)

    def _make_sim_state(
        self,
        position_world: np.ndarray,
        velocity_world: np.ndarray,
        angles: np.ndarray,
        rates: np.ndarray,
        motor_legacy: np.ndarray,
        target: np.ndarray,
        external_moment: np.ndarray,
    ) -> np.ndarray:
        state_world = np.zeros(19, dtype=np.float64)
        state_world[0:3] = position_world - target
        state_world[3:6] = velocity_world
        state_world[6:9] = angles
        state_world[9:12] = rates
        state_world[12:15] = external_moment
        state_world[15:19] = self._legacy_motor_to_omega(motor_legacy)
        return world_to_body_state(state_world)

    def _sim_to_world_state(self, sim_state: np.ndarray, target: np.ndarray) -> np.ndarray:
        state_world_rel = body_to_world_state(sim_state)
        out = np.zeros(16, dtype=np.float64)
        out[0:3] = target + state_world_rel[0:3]
        out[3:12] = state_world_rel[3:12]
        out[12:16] = self._omega_to_legacy_motor(state_world_rel[15:19])
        return out

    def _refresh_world_states(self) -> None:
        targets = self.gate_pos[self.target_gates % self.num_gates].astype(np.float64)
        self.world_states = np.asarray(
            [self._sim_to_world_state(state, target) for state, target in zip(self.sim_states, targets, strict=True)],
            dtype=np.float32,
        )

    def update_states_gate(self) -> None:
        self._refresh_world_states()
        gate_pos = self.gate_pos[self.target_gates % self.num_gates]
        gate_yaw = self.gate_yaw[self.target_gates % self.num_gates]
        rotation = np.array(
            [
                [np.cos(gate_yaw), np.sin(gate_yaw)],
                [-np.sin(gate_yaw), np.cos(gate_yaw)],
            ],
            dtype=np.float32,
        ).transpose((2, 1, 0))

        new_states = np.zeros((self.num_envs, self.state_len), dtype=np.float32)
        pos_w = self.world_states[:, 0:3]
        pos_g = (pos_w[:, np.newaxis, 0:2] - gate_pos[:, np.newaxis, 0:2]) @ rotation
        new_states[:, 0:2] = pos_g[:, 0, :]
        new_states[:, 2] = pos_w[:, 2] - gate_pos[:, 2]

        vel_w = self.world_states[:, 3:6]
        vel_g = vel_w[:, np.newaxis, 0:2] @ rotation
        new_states[:, 3:5] = vel_g[:, 0, :]
        new_states[:, 5] = vel_w[:, 2]

        new_states[:, 6:8] = self.world_states[:, 6:8]
        yaw = self.world_states[:, 8] - gate_yaw
        yaw %= 2.0 * np.pi
        yaw[yaw > np.pi] -= 2.0 * np.pi
        yaw[yaw < -np.pi] += 2.0 * np.pi
        new_states[:, 8] = yaw
        new_states[:, 9:16] = self.world_states[:, 9:16]

        for ahead_idx in range(self.gates_ahead):
            indices = (self.target_gates + ahead_idx + 1) % self.num_gates
            start = 16 + 4 * ahead_idx
            new_states[:, start:start + 3] = self.gate_pos_rel[indices]
            new_states[:, start + 3] = self.gate_yaw_rel[indices]

        self.action_hist = np.roll(self.action_hist, 1, axis=1)
        self.action_hist[:, 0] = self.actions
        action_offset = 16 + 4 * self.gates_ahead
        for hist_idx in range(self.num_action_history):
            start = action_offset + 4 * hist_idx
            source_idx = (hist_idx + 1) * self.history_step_size - 1
            new_states[:, start:start + 4] = self.action_hist[:, source_idx]

        # if self.param_input:
        #     new_states[:, action_offset + 4 * self.num_action_history:] = self.param_encoding

        if self.param_input_noise > 0.0:
            new_states[:, 9:12] += np.random.normal(loc=0.0, scale=self.param_input_noise,
                size=(self.num_envs, 3)).astype(np.float32)

        self.state_hist = np.roll(self.state_hist, 1, axis=1)
        self.state_hist[:, 0] = new_states
        self.states = self.state_hist[
            :,
            0:(self.num_state_history + 1) * self.history_step_size:self.history_step_size,
        ].reshape((self.num_envs, -1))
        if self.low_obs:
            self.states = np.concatenate([self.states[:, 0:3], self.states[:, 6:9], self.states[:, 12:]], axis=1)
        elif self.no_vel:
            self.states = np.concatenate([self.states[:, 0:3], self.states[:, 6:]], axis=1)
        elif self.no_ang_vel:
            self.states = np.concatenate([self.states[:, 0:9], self.states[:, 12:]], axis=1)

    def _reset_episode_metrics(self, dones: np.ndarray) -> None:
        self.episode_distance[dones] = 0.0
        self.episode_speed_sum[dones] = 0.0
        self.episode_speed_max[dones] = 0.0
        self.episode_bank_angle_sum[dones] = 0.0
        self.episode_action_sum[dones] = 0.0
        self.episode_action_sq_sum[dones] = 0.0
        self.episode_action_count[dones] = 0
        self.episode_action_rpm_diff_sum[dones] = 0.0
        self.episode_metric_steps[dones] = 0

    def _bank_angle_from_states(self, states: np.ndarray) -> np.ndarray:
        phi = states[:, 6]
        theta = states[:, 7]
        return np.arccos(np.clip(np.cos(phi) * np.cos(theta), -1.0, 1.0))

    def _action_to_commanded_speed_rpm(self, actions: np.ndarray) -> np.ndarray:
        info = quadrotor_sim_matlab.INFO
        return info.omega_min + self.action_to_bebop_command(actions) * (info.omega_max - info.omega_min)

    def reset_(self, dones: np.ndarray) -> np.ndarray:
        num_reset = int(dones.sum())
        if num_reset == 0:
            return self.states

        if self.initialize_at_random_gates:
            self.target_gates[dones] = np.random.randint(0, self.num_gates, size=num_reset)
            pos = self.gate_pos[self.target_gates[dones] % self.num_gates].astype(np.float64)
            yaw = self.gate_yaw[self.target_gates[dones] % self.num_gates].astype(np.float64)
            pos = pos - np.array([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)]).T
            x0, y0, z0 = pos.T
            x0 += np.random.uniform(-0.9, 0.9, size=num_reset)
            y0 += np.random.uniform(-0.9, 0.9, size=num_reset)
            z0 += np.random.uniform(-0.9, 0.9, size=num_reset)
        elif self.initialize_uniform:
            x0 = np.random.uniform(-5.0, 5.0, size=num_reset)
            y0 = np.random.uniform(-5.0, 5.0, size=num_reset)
            z0 = np.random.uniform(-3.0, 0.0, size=num_reset)
            dist_to_gate = np.zeros((num_reset, self.num_gates), dtype=np.float64)
            behind_gate = np.zeros((num_reset, self.num_gates), dtype=bool)
            for idx in range(self.num_gates):
                pos = self.gate_pos[idx]
                yaw = self.gate_yaw[idx]
                dist_to_gate[:, idx] = np.linalg.norm(np.stack([x0 - pos[0], y0 - pos[1]], axis=1), axis=1)
                behind_gate[:, idx] = np.cos(yaw) * (x0 - pos[0]) + np.sin(yaw) * (y0 - pos[1]) < 0.0
            closest_gate = np.zeros(num_reset, dtype=np.int64)
            for env_idx in range(num_reset):
                if not behind_gate[env_idx].any():
                    closest_gate[env_idx] = int(np.argmin(dist_to_gate[env_idx]))
                else:
                    behind_indices = np.where(behind_gate[env_idx])[0]
                    closest_gate[env_idx] = int(behind_indices[np.argmin(dist_to_gate[env_idx][behind_indices])])
            self.target_gates[dones] = closest_gate
        else:
            self.target_gates[dones] = self.start_gate
            jitter = self.start_pos_jitter
            x0 = np.random.uniform(-jitter, jitter, size=num_reset) + self.start_pos[0]
            y0 = np.random.uniform(-jitter, jitter, size=num_reset) + self.start_pos[1]
            z0 = np.random.uniform(-jitter, jitter, size=num_reset) + self.start_pos[2]

        velocity = np.stack(
            [
                np.random.uniform(-0.5, 0.5, size=num_reset) / 5.0,
                np.random.uniform(-0.5, 0.5, size=num_reset) / 5.0,
                np.random.uniform(-0.5, 0.5, size=num_reset) / 5.0,
            ],
            axis=1,
        )
        angles = np.stack(
            [
                np.random.uniform(-np.pi / 9.0, np.pi / 9.0, size=num_reset) / 4.0,
                np.random.uniform(-np.pi / 9.0, np.pi / 9.0, size=num_reset) / 4.0,
                np.random.uniform(-np.pi, np.pi, size=num_reset),
            ],
            axis=1,
        )
        rates = np.stack(
            [
                np.random.uniform(-0.1, 0.1, size=num_reset) / 5.0,
                np.random.uniform(-0.1, 0.1, size=num_reset) / 5.0,
                np.random.uniform(-0.1, 0.1, size=num_reset) / 5.0,
            ],
            axis=1,
        )
        motors = np.stack(
            [
                np.random.uniform(-1.0, 1.0, size=num_reset) / 2.0,
                np.random.uniform(-1.0, 1.0, size=num_reset) / 2.0,
                np.random.uniform(-1.0, 1.0, size=num_reset) / 2.0,
                np.random.uniform(-1.0, 1.0, size=num_reset) / 2.0,
            ],
            axis=1,
        )
        positions = np.stack([x0, y0, z0], axis=1)
        targets = self.gate_pos[self.target_gates[dones] % self.num_gates].astype(np.float64)
        external_moments = self._sample_external_moments(num_reset)
        self.sim_states[dones] = np.asarray(
            [
                self._make_sim_state(position, velocity_i, angles_i, rates_i, motors_i, target, moment)
                for position, velocity_i, angles_i, rates_i, motors_i, target, moment in zip(
                    positions,
                    velocity,
                    angles,
                    rates,
                    motors,
                    targets,
                    external_moments,
                    strict=True,
                )
            ],
            dtype=np.float32,
        )
        self.step_counts[dones] = 0
        self.prev_derivs = [None if done else deriv for done, deriv in zip(dones, self.prev_derivs, strict=True)]
        self.state_hist[dones] = 0.0
        self.action_hist[dones] = 0.0
        self._reset_episode_metrics(dones)
        self.update_states_gate()
        return self.states

    def reset(self) -> np.ndarray:
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def step_async(self, actions) -> None:
        self.prev_actions = self.actions
        if self.action_range == "neg1_1":
            low = -1.0
            high = 2.0 * self.motor_limit - 1.0
        else:
            low = 0.0
            high = self.motor_limit
        self.actions = np.clip(np.asarray(actions, dtype=np.float32), low, high)

    def step_wait(self):
        old_sim_states = self.sim_states.astype(np.float64)
        old_targets = self.gate_pos[self.target_gates % self.num_gates].astype(np.float64)
        old_world_rel = np.asarray([body_to_world_state(state) for state in old_sim_states], dtype=np.float64)
        pos_old = old_targets + old_world_rel[:, 0:3]

        actions01 = self.action_to_bebop_command(self.actions.astype(np.float64))
        new_sim_states = np.zeros_like(old_sim_states)
        new_derivs: list[np.ndarray | None] = [None] * self.num_envs
        for idx in range(self.num_envs):
            next_state, deriv = integrate_state(
                self.integration_method,
                old_sim_states[idx],
                actions01[idx],
                self.dt,
                prev_deriv=self.prev_derivs[idx],
                implicit_iters=self.implicit_iters,
            )
            new_sim_states[idx] = next_state
            new_derivs[idx] = deriv

        self.step_counts += 1
        new_world_rel = np.asarray([body_to_world_state(state) for state in new_sim_states], dtype=np.float64)
        pos_new = old_targets + new_world_rel[:, 0:3]
        vel_new = new_world_rel[:, 3:6]

        step_distance = np.linalg.norm(pos_new - pos_old, axis=1)
        step_speed = np.linalg.norm(vel_new, axis=1)
        step_bank_angle = self._bank_angle_from_states(new_world_rel)
        commanded_motor_speed_rpm = self._action_to_commanded_speed_rpm(self.actions.astype(np.float64))
        actual_motor_speed_rpm = new_world_rel[:, 15:19]
        step_action_rpm_diff = np.mean(np.abs(commanded_motor_speed_rpm - actual_motor_speed_rpm), axis=1)

        self.episode_distance += step_distance
        self.episode_speed_sum += step_speed
        self.episode_speed_max = np.maximum(self.episode_speed_max, step_speed)
        self.episode_bank_angle_sum += step_bank_angle
        self.episode_action_sum += np.sum(self.actions, axis=1)
        self.episode_action_sq_sum += np.sum(self.actions**2, axis=1)
        self.episode_action_count += self.actions.shape[1]
        self.episode_action_rpm_diff_sum += step_action_rpm_diff
        self.episode_metric_steps += 1

        pos_gate = self.gate_pos[self.target_gates % self.num_gates].astype(np.float64)
        yaw_gate = self.gate_yaw[self.target_gates % self.num_gates].astype(np.float64)
        d2g_old = np.linalg.norm(pos_old - pos_gate, axis=1)
        d2g_new = np.linalg.norm(pos_new - pos_gate, axis=1)
        rate_penalty = 0.001 * np.linalg.norm(new_world_rel[:, 9:12], axis=1)
        rewards = d2g_old - d2g_new - rate_penalty

        normal = np.array([np.cos(yaw_gate), np.sin(yaw_gate)]).T
        pos_old_projected = (pos_old[:, 0] - pos_gate[:, 0]) * normal[:, 0] + (
            pos_old[:, 1] - pos_gate[:, 1]
        ) * normal[:, 1]
        pos_new_projected = (pos_new[:, 0] - pos_gate[:, 0]) * normal[:, 0] + (
            pos_new[:, 1] - pos_gate[:, 1]
        ) * normal[:, 1]
        passed_gate_plane = (pos_old_projected < 0.0) & (pos_new_projected > 0.0)
        gate_passed = passed_gate_plane & np.all(np.abs(pos_new - pos_gate) < self.gate_size / 2.0, axis=1)
        gate_collision = passed_gate_plane & np.any(np.abs(pos_new - pos_gate) > self.gate_size / 2.0, axis=1)

        ground_collision = pos_new[:, 2] > 0.0
        out_of_bounds = np.any(np.abs(pos_new[:, 0:2]) > 5.0, axis=1)
        out_of_bounds |= pos_new[:, 2] < -7.0
        out_of_bounds |= np.any(np.abs(new_world_rel[:, 9:12]) > 1000.0, axis=1)
        rewards[ground_collision] = -10.0
        rewards[out_of_bounds] = -10.0
        max_steps_reached = self.step_counts >= self.max_steps

        self.target_gates[gate_passed] += 1
        self.target_gates %= self.num_gates
        if np.any(gate_passed):
            updated_targets = self.gate_pos[self.target_gates % self.num_gates].astype(np.float64)
            for idx in np.where(gate_passed)[0]:
                relative_world = new_world_rel[idx].copy()
                relative_world[0:3] = pos_new[idx] - updated_targets[idx]
                new_sim_states[idx] = world_to_body_state(relative_world)

        dones = max_steps_reached | ground_collision | out_of_bounds | gate_collision
        self.dones = dones

        infos = [{} for _ in range(self.num_envs)]
        for idx in range(self.num_envs):
            if dones[idx]:
                metric_steps = max(int(self.episode_metric_steps[idx]), 1)
                action_count = max(int(self.episode_action_count[idx]), 1)
                action_mean = self.episode_action_sum[idx] / action_count
                action_var = max(self.episode_action_sq_sum[idx] / action_count - action_mean**2, 0.0)
                infos[idx]["terminal_observation"] = self.states[idx]
                infos[idx]["TimeLimit.time"] = int(self.step_counts[idx])
                infos[idx]["distance_travelled"] = float(self.episode_distance[idx])
                infos[idx]["max_speed"] = float(self.episode_speed_max[idx])
                infos[idx]["avg_speed"] = float(self.episode_speed_sum[idx] / metric_steps)
                infos[idx]["avg_bank_angle"] = float(self.episode_bank_angle_sum[idx] / metric_steps)
                infos[idx]["avg_action_command"] = float(action_mean)
                infos[idx]["action_std"] = float(np.sqrt(action_var))
                infos[idx]["avg_action_rpm_diff"] = float(self.episode_action_rpm_diff_sum[idx] / metric_steps)
            infos[idx]["TimeLimit.truncated"] = bool(max_steps_reached[idx])
            infos[idx]["ground_collision"] = bool(ground_collision[idx])
            infos[idx]["out_of_bounds"] = bool(out_of_bounds[idx])
            infos[idx]["gate_collision"] = bool(gate_collision[idx])
            infos[idx]["gate_passed"] = bool(gate_passed[idx])
            infos[idx]["target_gate"] = int(self.target_gates[idx])
            infos[idx]["distance_to_gate"] = float(d2g_new[idx])
            infos[idx]["rewards"] = float(rewards[idx])

        self.sim_states = new_sim_states.astype(np.float32)
        self.prev_derivs = new_derivs
        self.update_states_gate()
        self.reset_(dones)
        return self.states, rewards.astype(np.float32), dones, infos

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed_value = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
        return [seed]

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            return [getattr(self, attr_name) for _ in range(self.num_envs)]
        if isinstance(indices, int):
            indices = [indices]
        return [getattr(self, attr_name) for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        method = getattr(self, method_name)
        return [method(*method_args, **method_kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def render(self, mode="human"):
        self._refresh_world_states()
        keys = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r", "w1", "w2", "w3", "w4"]
        state_dict = dict(zip(keys, self.world_states.T, strict=True))
        action_dict = dict(zip(["u1", "u2", "u3", "u4"], self.action_to_bebop_command(self.actions).T, strict=True))
        return {**state_dict, **action_dict}
