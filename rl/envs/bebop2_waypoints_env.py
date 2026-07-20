#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vectorized Bebop2 waypoint environment for PPO/SB3 experiments.

This is the first non-legacy RL environment for this repository.  It keeps the
same controller interface as the supervised-learning models:

- observation: 19-state vector
- action: four normalized motor commands in [0, 1]
- dynamics: utils.dynamics_models.quadrotor_sim_matlab
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from ..bc_policy import FrozenBCController
from ...utils.data import get_norm_vectors
from ...utils.dynamics_models import quadrotor_sim_matlab, set_dynamics_model
from ...utils.quadrotor_sim import STATE_LABELS, body_to_world_state, integrate_state, world_to_body_state


DEFAULT_SQUARE_WAYPOINTS = np.asarray(
    [
        [2.0, 1.5, -1.5],
        [2.0, -1.5, -1.5],
        [-2.0, -1.5, -1.5],
        [-2.0, 1.5, -1.5],
    ],
    dtype=np.float32,
)
DEFAULT_START_POS = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)


class Bebop2WaypointEnv(VecEnv):
    """SB3 VecEnv that tracks point waypoints with the Bebop2 MATLAB dynamics."""

    def __init__(
        self,
        num_envs: int,
        waypoints: np.ndarray | Sequence[Sequence[float]] = DEFAULT_SQUARE_WAYPOINTS,
        start_pos: np.ndarray | Sequence[float] = DEFAULT_START_POS,
        waypoint_radius: float = 0.20,
        dt: float = 0.01,
        max_steps: int = 6000,
        integration_method: str = "rk4",
        implicit_iters: int = 1,
        initialize_at_random_waypoints: bool = False,
        terminate_on_waypoint: bool = True,
        normalize_observations: bool = False,
        normalization_limits: str = "bebop2_tau_0_06",
        randomize_external_moments: bool = True,
        seed: int | None = None,
    ):
        self.seed(seed)
        set_dynamics_model("quadrotor_sim_matlab")

        self.waypoints = np.asarray(waypoints, dtype=np.float32)
        self.start_pos = np.asarray(start_pos, dtype=np.float32)
        self.num_waypoints = int(self.waypoints.shape[0])
        self.waypoint_radius = float(waypoint_radius)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.integration_method = integration_method
        self.implicit_iters = int(implicit_iters)
        self.initialize_at_random_waypoints = initialize_at_random_waypoints
        self.terminate_on_waypoint = bool(terminate_on_waypoint)
        self.normalize_observations = bool(normalize_observations)
        self.randomize_external_moments = bool(randomize_external_moments)
        norm_min, norm_max = get_norm_vectors(STATE_LABELS, normalization_limits)
        self.obs_min = norm_min.reshape(-1).astype(np.float32)
        self.obs_max = norm_max.reshape(-1).astype(np.float32)

        action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        observation_space = spaces.Box(
            low=np.asarray([-np.inf] * 19, dtype=np.float32),
            high=np.asarray([np.inf] * 19, dtype=np.float32),
            shape=(19,),
            dtype=np.float32,
        )
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        self.states = np.zeros((num_envs, 19), dtype=np.float32)
        self.actions = np.zeros((num_envs, 4), dtype=np.float32)
        self.prev_actions = np.zeros((num_envs, 4), dtype=np.float32)
        self.prev_derivs: list[np.ndarray | None] = [None] * num_envs
        self.target_waypoints = np.zeros(num_envs, dtype=np.int64)
        self.step_counts = np.zeros(num_envs, dtype=np.int64)

        self.episode_distance = np.zeros(num_envs, dtype=np.float64)
        self.episode_speed_sum = np.zeros(num_envs, dtype=np.float64)
        self.episode_speed_max = np.zeros(num_envs, dtype=np.float64)
        self.episode_action_sq_sum = np.zeros(num_envs, dtype=np.float64)
        self.episode_action_count = np.zeros(num_envs, dtype=np.int64)
        self.episode_waypoints = np.zeros(num_envs, dtype=np.int64)

        self.reset()

    def _normalize_states(self, states: np.ndarray) -> np.ndarray:
        denom = self.obs_max - self.obs_min + 1e-10
        return ((states - self.obs_min) / denom).astype(np.float32)

    def _observations(self) -> np.ndarray:
        if not self.normalize_observations:
            return self.states
        return self._normalize_states(self.states)

    def reset_seed(self):
        if self._seed_value is not None:
            np.random.seed(self._seed_value)
            torch.manual_seed(self._seed_value)

    def _target_positions(self) -> np.ndarray:
        return self.waypoints[self.target_waypoints % self.num_waypoints]

    def _world_relative_state(self, state_body: np.ndarray) -> np.ndarray:
        return body_to_world_state(state_body)

    def _absolute_position(self, state_body: np.ndarray, target: np.ndarray) -> np.ndarray:
        return target + self._world_relative_state(state_body)[0:3]

    def _sample_external_moments(self, num_samples: int) -> np.ndarray:
        if not self.randomize_external_moments:
            return np.zeros((num_samples, 3), dtype=np.float64)
        return np.random.uniform(
            low=self.obs_min[12:15],
            high=self.obs_max[12:15],
            size=(num_samples, 3),
        ).astype(np.float64)

    def _make_body_state(
        self,
        absolute_position: np.ndarray,
        target: np.ndarray,
        external_moment: np.ndarray,
    ) -> np.ndarray:
        state_world = np.zeros(19, dtype=np.float64)
        state_world[0:3] = absolute_position - target
        state_world[12:15] = external_moment
        hover_omega = quadrotor_sim_matlab.INFO.hover_omega
        state_world[15:19] = hover_omega if hover_omega is not None else quadrotor_sim_matlab.INFO.omega_mid
        return world_to_body_state(state_world)

    def _reset_episode_metrics(self, dones: np.ndarray) -> None:
        self.episode_distance[dones] = 0.0
        self.episode_speed_sum[dones] = 0.0
        self.episode_speed_max[dones] = 0.0
        self.episode_action_sq_sum[dones] = 0.0
        self.episode_action_count[dones] = 0
        self.episode_waypoints[dones] = 0

    def reset_(self, dones: np.ndarray) -> np.ndarray:
        num_reset = int(dones.sum())
        if num_reset == 0:
            return self._observations()

        if self.initialize_at_random_waypoints:
            self.target_waypoints[dones] = np.random.randint(0, self.num_waypoints, size=num_reset)
            target = self._target_positions()[dones]
            offset = np.random.uniform(-0.6, 0.6, size=(num_reset, 3))
            offset[:, 2] *= 0.5
            absolute_positions = target + offset
        else:
            self.target_waypoints[dones] = 0
            absolute_positions = self.start_pos + np.random.uniform(-0.5, 0.5, size=(num_reset, 3))
            # absolute_positions[:, 2] = self.start_pos[2] + np.random.uniform(-0.15, 0.15, size=num_reset)
            absolute_positions[:, :2] = self.start_pos[:2] + np.random.uniform(-3, 3, size=(num_reset, 2))

        targets = self._target_positions()[dones]
        external_moments = self._sample_external_moments(num_reset)
        reset_states = [
            self._make_body_state(absolute_position, target, external_moment)
            for absolute_position, target, external_moment in zip(
                absolute_positions,
                targets,
                external_moments,
                strict=True,
            )
        ]
        self.states[dones] = np.asarray(reset_states, dtype=np.float32)
        self.step_counts[dones] = 0
        self.prev_derivs = [None if done else deriv for done, deriv in zip(dones, self.prev_derivs, strict=True)]
        self._reset_episode_metrics(dones)
        return self._observations()

    def reset(self) -> np.ndarray:
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def step_async(self, actions) -> None:
        self.prev_actions = self.actions
        self.actions = np.clip(np.asarray(actions, dtype=np.float32), 0.0, 1.0)

    def step_wait(self):
        old_states = self.states.astype(np.float64)
        actions = self.actions.astype(np.float64)

        new_states = np.zeros_like(old_states)
        new_derivs: list[np.ndarray | None] = [None] * self.num_envs
        for idx in range(self.num_envs):
            next_state, deriv = integrate_state(
                self.integration_method,
                old_states[idx],
                actions[idx],
                self.dt,
                prev_deriv=self.prev_derivs[idx],
                implicit_iters=self.implicit_iters,
            )
            new_states[idx] = next_state
            new_derivs[idx] = deriv

        self.step_counts += 1
        targets = self._target_positions().astype(np.float64)
        old_world = np.asarray([self._world_relative_state(state) for state in old_states], dtype=np.float64)
        new_world = np.asarray([self._world_relative_state(state) for state in new_states], dtype=np.float64)
        old_abs = targets + old_world[:, 0:3]
        new_abs = targets + new_world[:, 0:3]

        d2w_old = np.linalg.norm(old_abs - targets, axis=1)
        d2w_new = np.linalg.norm(new_abs - targets, axis=1)
        waypoint_reached = d2w_new < self.waypoint_radius

        speed = np.linalg.norm(new_states[:, 3:6], axis=1)
        rate_penalty = np.linalg.norm(new_states[:, 9:12], axis=1)
        angle_penalty = np.linalg.norm(new_states[:, 6:8], axis=1)
        # action_penalty = 0.0 * np.linalg.norm(actions, axis=1)
        action_penalty_delta = 0.001 * np.linalg.norm(actions - self.prev_actions, axis=1)
        progress_reward = d2w_old - d2w_new
        # max_speed = 12.0
        # cap progress rewards to be less than max_speed*dt
        # progress_reward[progress_reward > max_speed * self.dt] = max_speed * self.dt

        # rewards = progress_reward - rate_penalty - angle_penalty
        rewards = progress_reward - 0.001 * rate_penalty - action_penalty_delta # - action_penalty - action_penalty_delta

        # Waypoint reward + dist penalty
        in_hover_region = (
            (d2w_new < self.waypoint_radius) &
            (angle_penalty < 0.3) &
            (speed < 0.25)
        )
        # rewards[waypoint_reached] = 1.0 # CHANGED HERE, WAS COMMENTED BEFORE SO WATCH OUT ----------
        rewards[in_hover_region] = 1.0 # CHANGED HERE, WAS COMMENTED BEFORE SO WATCH OUT ----------

        step_distance = np.linalg.norm(new_abs - old_abs, axis=1)
        self.episode_distance += step_distance
        self.episode_speed_sum += speed
        self.episode_speed_max = np.maximum(self.episode_speed_max, speed)
        self.episode_action_sq_sum += np.sum(actions**2, axis=1)
        self.episode_action_count += actions.shape[1]
        self.episode_waypoints += waypoint_reached.astype(np.int64)

        continue_to_next_waypoint = waypoint_reached & (not self.terminate_on_waypoint)
        self.target_waypoints[continue_to_next_waypoint] += 1
        self.target_waypoints %= self.num_waypoints

        # Recenter the 19-state vector relative to the new target waypoint only
        # for continuous square rollouts. Training episodes end at the reached point.
        if np.any(continue_to_next_waypoint):
            updated_targets = self._target_positions().astype(np.float64)
            for idx in np.where(continue_to_next_waypoint)[0]:
                relative_world = new_world[idx].copy()
                relative_world[0:3] = new_abs[idx] - updated_targets[idx]
                new_states[idx] = world_to_body_state(relative_world)

        ground_collision = new_abs[:, 2] > 0.0
        out_of_bounds = np.any(np.abs(new_abs[:, 0:2]) > 6.0, axis=1)
        out_of_bounds |= new_abs[:, 2] < -7.0
        out_of_bounds |= np.any(np.abs(new_states[:, 9:12]) > 1000.0, axis=1)
        max_steps_reached = self.step_counts >= self.max_steps

        rewards[ground_collision] = -10.0
        rewards[out_of_bounds] = -10.0
        dones = ground_collision | out_of_bounds | max_steps_reached
        if self.terminate_on_waypoint:
            dones |= waypoint_reached

        infos = [{} for _ in range(self.num_envs)]
        for idx in range(self.num_envs):
            if dones[idx]:
                metric_steps = max(int(self.step_counts[idx]), 1)
                action_count = max(int(self.episode_action_count[idx]), 1)
                terminal_state = new_states[idx:idx + 1].astype(np.float32)
                infos[idx]["terminal_observation"] = (
                    self._normalize_states(terminal_state)[0]
                    if self.normalize_observations
                    else terminal_state[0]
                )
                infos[idx]["TimeLimit.time"] = int(self.step_counts[idx])
                infos[idx]["TimeLimit.truncated"] = bool(max_steps_reached[idx])
                infos[idx]["distance_travelled"] = float(self.episode_distance[idx])
                infos[idx]["max_speed"] = float(self.episode_speed_max[idx])
                infos[idx]["avg_speed"] = float(self.episode_speed_sum[idx] / metric_steps)
                infos[idx]["avg_action_rms"] = float(np.sqrt(self.episode_action_sq_sum[idx] / action_count))
                infos[idx]["waypoints_reached"] = int(self.episode_waypoints[idx])
            else:
                infos[idx]["TimeLimit.truncated"] = False
            infos[idx]["ground_collision"] = bool(ground_collision[idx])
            infos[idx]["out_of_bounds"] = bool(out_of_bounds[idx])
            infos[idx]["waypoint_reached"] = bool(waypoint_reached[idx])
            infos[idx]["target_waypoint"] = int(self.target_waypoints[idx])
            infos[idx]["distance_to_waypoint"] = float(d2w_new[idx])
            infos[idx]["rewards"] = float(rewards[idx])

        self.states = new_states.astype(np.float32)
        self.prev_derivs = new_derivs
        self.reset_(dones)
        return self._observations(), rewards.astype(np.float32), dones, infos

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
        targets = self._target_positions().astype(np.float64)
        abs_positions = np.asarray(
            [self._absolute_position(state, target) for state, target in zip(self.states, targets, strict=True)],
            dtype=np.float64,
        )
        return {
            "x": abs_positions[:, 0],
            "y": abs_positions[:, 1],
            "z": abs_positions[:, 2],
            "phi": self.states[:, 6],
            "theta": self.states[:, 7],
            "psi": self.states[:, 8],
            "u1": self.actions[:, 0],
            "u2": self.actions[:, 1],
            "u3": self.actions[:, 2],
            "u4": self.actions[:, 3],
        }


class ResidualBebop2WaypointEnv(Bebop2WaypointEnv):
    """Waypoint env where PPO actions are residual corrections around a frozen SL controller."""

    def __init__(
        self,
        num_envs: int,
        bc_config_path: str | Path,
        bc_checkpoint_path: str | Path,
        project_root: str | Path,
        residual_scale: float = 0.05,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(num_envs=num_envs, **kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.base_controller = FrozenBCController(
            bc_config_path=bc_config_path,
            bc_checkpoint_path=bc_checkpoint_path,
            project_root=project_root,
            device=device,
        )
        self.residual_scale = float(residual_scale)
        self.base_actions = np.zeros((num_envs, 4), dtype=np.float32)
        self.residual_actions = np.zeros((num_envs, 4), dtype=np.float32)

    def step_async(self, actions) -> None:
        self.residual_actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        self.base_actions = self.base_controller.predict_numpy(self.states).astype(np.float32)
        final_actions = np.clip(
            self.base_actions + self.residual_scale * self.residual_actions,
            0.0,
            1.0,
        )
        super().step_async(final_actions)
