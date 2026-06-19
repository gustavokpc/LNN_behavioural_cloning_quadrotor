#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train PPO/SB3 on the Bebop2 waypoint environment.

This is the non-legacy training entrypoint. It uses:

- Bebop2WaypointEnv, not Quadcopter3DGates
- 19 observations, matching the supervised-learning input interface
- 4 actions in [0, 1], matching the supervised-learning output convention
- quadrotor_sim_matlab dynamics through the environment
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO

from .bc_policy import BCInitializedActorCriticPolicy
from .envs.bebop2_figure8_gates_env import Bebop2Figure8GatesEnv
from .envs.bebop2_waypoints_env import DEFAULT_SQUARE_WAYPOINTS, Bebop2WaypointEnv, ResidualBebop2WaypointEnv
from .legacy_ppo.quadcopter_animation import animation as legacy_animation
from .legacy_ppo.drone_ppo_sb3 import (
    GradientEpisodePrintCallback,
    attach_gradient_logger,
    resolve_algorithm,
)
from ..utils.animation import animate
from ..utils.quadrotor_sim import body_to_world_state


RL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RL_ROOT.parent


def _track_artifact_name(track: str) -> str:
    return "bebop2_waypoints" if track == "square_waypoints" else track


def _state_to_absolute_position(state: np.ndarray, target: np.ndarray) -> np.ndarray:
    return target + body_to_world_state(state)[0:3]


def make_env(
    args: argparse.Namespace,
    num_envs: int,
    seed: int | None = None,
    terminate_on_waypoint: bool | None = None,
):
    if args.track == "figure8_gates":
        if args.policy_type in {"bc_ppo", "residual_ppo"}:
            raise ValueError("--track figure8_gates uses legacy-style observations; choose --policy-type ppo or recurrent_ppo.")
        return Bebop2Figure8GatesEnv(
            num_envs=num_envs,
            gates_ahead=args.gates_ahead,
            gate_size=args.gate_size,
            dt=args.dt,
            max_steps=args.max_steps,
            integration_method=args.integration_method,
            implicit_iters=args.implicit_iters,
            initialize_at_random_gates=args.initialize_at_random_gates,
            initialize_uniform=args.initialize_uniform,
            num_state_history=args.num_state_history,
            num_action_history=args.num_action_history,
            history_step_size=args.history_step_size,
            param_input=args.param_input,
            param_input_noise=args.param_input_noise,
            low_obs=args.low_obs,
            no_vel=args.no_vel,
            no_ang_vel=args.no_ang_vel,
            randomize_external_moments=args.randomize_external_moments,
            seed=seed,
        )
    if terminate_on_waypoint is None:
        terminate_on_waypoint = True
    normalize_observations = bool(args.normalize_observations and args.policy_type in {"ppo", "recurrent_ppo"})
    env_kwargs = dict(
        num_envs=num_envs,
        waypoint_radius=args.waypoint_radius,
        dt=args.dt,
        max_steps=args.max_steps,
        integration_method=args.integration_method,
        implicit_iters=args.implicit_iters,
        initialize_at_random_waypoints=args.initialize_at_random_waypoints,
        terminate_on_waypoint=terminate_on_waypoint,
        normalize_observations=normalize_observations,
        randomize_external_moments=args.randomize_external_moments,
        seed=seed,
    )
    if args.policy_type == "residual_ppo":
        return ResidualBebop2WaypointEnv(
            **env_kwargs,
            bc_config_path=args.bc_config,
            bc_checkpoint_path=args.bc_checkpoint,
            project_root=PROJECT_ROOT,
            residual_scale=args.residual_scale,
            device=args.device if args.device != "auto" else "cpu",
        )
    return Bebop2WaypointEnv(**env_kwargs)


def train(args: argparse.Namespace) -> None:
    algo_cls, policy_class, policy_kwargs = resolve_bebop2_algorithm(args)
    algo_tag = args.policy_type
    track_artifact = _track_artifact_name(args.track)
    ckpt_dir = RL_ROOT / "checkpoints" / track_artifact / algo_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(args, num_envs=args.num_envs, seed=args.seed)
    env = VecMonitor(train_env)

    if args.cont:
        model_path = Path(args.cont)
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint '{model_path}' not found.")
        model = algo_cls.load(model_path, env=env, device=args.device, custom_objects={"policy_class": policy_class})
    else:
        algo_kwargs: dict[str, Any] = {}
        model = algo_cls(
            policy_class,
            env,
            learning_rate=args.learning_rate,
            n_steps=args.rollout_fragment_length,
            batch_size=args.batch_size,
            ent_coef=args.entropy_coeff,
            clip_range=args.clip_param,
            vf_coef=args.vf_coeff,
            gamma=args.gamma,
            gae_lambda=args.lam,
            tensorboard_log=args.tensorboard_log,
            n_epochs=args.n_epochs,
            verbose=1,
            device=args.device,
            policy_kwargs=policy_kwargs,
            **algo_kwargs,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(ckpt_dir),
        name_prefix=f"{algo_tag}_{track_artifact}",
    )
    attach_gradient_logger(model.policy)
    grad_cb = GradientEpisodePrintCallback()

    start = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, grad_cb],
        progress_bar=True,
    )
    model.save(ckpt_dir / f"{algo_tag}_{track_artifact}")
    env.close()
    elapsed = (time.time() - start) / 3600
    print(f"Finished training {args.total_timesteps:,} steps in {elapsed:0.2f}h. Latest checkpoint saved to {ckpt_dir}.")


def _collect_render_episode(model, args: argparse.Namespace):
    env = make_env(args, num_envs=1, seed=args.seed, terminate_on_waypoint=False)
    obs = env.reset()
    state = None
    episode_start = np.ones((env.num_envs,), dtype=bool)
    positions = []
    orientations = []
    actions_history = []
    targets = []
    rewards = []

    for _ in range(args.render_steps):
        action, state = model.predict(
            obs,
            state=state,
            episode_start=episode_start,
            deterministic=True,
        )
        obs, reward, done, infos = env.step(action)
        target = env._target_positions()[0].astype(np.float64)
        positions.append(_state_to_absolute_position(env.states[0], target))
        orientations.append(env.states[0, 6:9].copy())
        actions_history.append(env.actions[0].astype(np.float64))
        targets.append(target.copy())
        rewards.append(float(reward[0]))
        reached_waypoint = bool(infos[0].get("waypoint_reached", False))
        episode_start = done | (args.reset_recurrent_at_waypoint and np.asarray([reached_waypoint], dtype=bool))
        if bool(done[0]):
            break

    if not positions:
        raise RuntimeError("No rollout samples were collected for rendering.")

    rollout = {
        "positions": np.asarray(positions, dtype=np.float64),
        "orientations": np.asarray(orientations, dtype=np.float64),
        "actions": np.asarray(actions_history, dtype=np.float64),
        "targets": np.asarray(targets, dtype=np.float64),
        "t": np.arange(len(positions), dtype=np.float64) * args.dt,
        "reward": float(sum(rewards)),
    }
    env.close()
    return rollout


def render(args: argparse.Namespace) -> None:
    if not args.cont:
        raise ValueError("--render requires --cont pointing to a trained SB3 checkpoint.")

    algo_cls, policy_class, _ = resolve_bebop2_algorithm(args)
    if args.track == "figure8_gates":
        env = make_env(args, num_envs=1, seed=args.seed)
        model = algo_cls.load(
            args.cont,
            env=env,
            device=args.device,
            custom_objects={"policy_class": policy_class},
        )
        obs = env.reset()
        state = None
        episode_start = np.ones((env.num_envs,), dtype=bool)

        def run():
            nonlocal obs, state, episode_start
            actions, state = model.predict(
                obs,
                state=state,
                episode_start=episode_start,
                deterministic=True,
            )
            obs, _, dones, _ = env.step(actions)
            episode_start = dones
            if dones.any():
                state = None
            return env.render()

        legacy_animation.view(
            run,
            gate_pos=env.gate_pos,
            gate_yaw=env.gate_yaw,
            fps=1 / env.dt,
            record_steps=args.render_steps if args.record else 0,
            record_file=args.output,
            show_window=args.auto_play or not args.record,
        )
        env.close()
        return

    load_env = make_env(args, num_envs=1, seed=args.seed)
    model = algo_cls.load(
        args.cont,
        env=load_env,
        device=args.device,
        custom_objects={"policy_class": policy_class},
    )
    load_env.close()

    rollouts = []
    for episode_idx in range(args.render_episodes):
        episode_args = argparse.Namespace(**vars(args))
        episode_args.seed = None if args.seed is None else args.seed + episode_idx
        rollout = _collect_render_episode(model, episode_args)
        rollouts.append(rollout)
        print(
            f"Episode {episode_idx + 1}: "
            f"steps={len(rollout['t'])} | total_reward={rollout['reward']:.3f}"
        )

    if args.render_episodes == 1:
        rollout = rollouts[0]
        positions_arr = rollout["positions"]
        orientations_arr = rollout["orientations"]
        actions_arr = rollout["actions"]
        targets_arr = rollout["targets"]
        t = rollout["t"]

        animate(
            t=t,
            x=positions_arr[:, 0],
            y=positions_arr[:, 1],
            z=positions_arr[:, 2],
            phi=orientations_arr[:, 0],
            theta=orientations_arr[:, 1],
            psi=orientations_arr[:, 2],
            u=actions_arr,
            target=targets_arr,
            waypoints=DEFAULT_SQUARE_WAYPOINTS.astype(np.float64),
            file=args.output,
            record=args.record,
            auto_play=args.auto_play or args.record,
            close_on_end=args.record,
            draw_path=True,
        )
        return

    animate(
        t=[rollout["t"] for rollout in rollouts],
        x=[rollout["positions"][:, 0] for rollout in rollouts],
        y=[rollout["positions"][:, 1] for rollout in rollouts],
        z=[rollout["positions"][:, 2] for rollout in rollouts],
        phi=[rollout["orientations"][:, 0] for rollout in rollouts],
        theta=[rollout["orientations"][:, 1] for rollout in rollouts],
        psi=[rollout["orientations"][:, 2] for rollout in rollouts],
        u=[rollout["actions"] for rollout in rollouts],
        target=[],
        waypoints=DEFAULT_SQUARE_WAYPOINTS.astype(np.float64),
        file=args.output,
        multiple_trajectories=True,
        simultaneous=args.simultaneous,
        names=[f"episode_{idx + 1}" for idx in range(len(rollouts))],
        record=args.record,
        auto_play=args.auto_play or args.record,
        close_on_end=args.record,
        draw_path=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-log-std", type=float, default=-1.5)
    parser.add_argument("--rollout-fragment-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.05)
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    parser.add_argument("--vf-coeff", type=float, default=0.5)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--tensorboard-log", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--cfc-timespan", type=float, default=0.01)
    parser.add_argument("--use-flatten-features", type=bool, default=True)
    parser.add_argument("--policy-type", choices=("ppo", "recurrent_ppo", "bc_ppo", "residual_ppo"), default="recurrent_ppo")
    parser.add_argument(
        "--bc-config",
        type=str,
        default=str(PROJECT_ROOT / "configs/new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.yaml"),
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=str,
        default="new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.ckpt",
    )
    parser.add_argument("--bc-value-hidden-dim", type=int, default=64)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--cont", type=str, default="")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--track", choices=("square_waypoints", "figure8_gates"), default="square_waypoints")
    parser.add_argument("--waypoint-radius", type=float, default=0.2)
    parser.add_argument("--gate-size", type=float, default=1.5)
    parser.add_argument("--gates-ahead", type=int, default=1)
    parser.add_argument("--integration-method", default="rk4")
    parser.add_argument("--implicit-iters", type=int, default=1)
    parser.add_argument("--initialize-at-random-waypoints", action="store_true")
    parser.add_argument("--initialize-at-random-gates", action="store_true")
    parser.add_argument("--initialize-uniform", action="store_true")
    parser.add_argument("--num-state-history", type=int, default=0)
    parser.add_argument("--num-action-history", type=int, default=0)
    parser.add_argument("--history-step-size", type=int, default=1)
    parser.add_argument("--low-obs", action="store_true")
    parser.add_argument("--no-vel", action="store_true")
    parser.add_argument("--no-ang-vel", action="store_true")
    parser.add_argument("--param-input", action="store_true")
    parser.add_argument("--param-input-noise", type=float, default=0.0)
    parser.add_argument("--normalize-observations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize-external-moments", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-steps", type=int, default=2000)
    parser.add_argument("--render-episodes", type=int, default=1)
    parser.add_argument("--reset-recurrent-at-waypoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simultaneous", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--auto-play", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-std-init", type=float, default=-3.0)
    args = parser.parse_args()
    track_artifact = _track_artifact_name(args.track)
    if not args.tensorboard_log:
        args.tensorboard_log = str(RL_ROOT / "runs" / track_artifact)
    if not args.output:
        args.output = str(RL_ROOT / "runs" / f"{track_artifact}_rollout.mp4")
    if args.track == "figure8_gates" and args.policy_type in {"bc_ppo", "residual_ppo"}:
        raise ValueError("--track figure8_gates uses legacy-style observations; choose --policy-type ppo or recurrent_ppo.")
    return args


def resolve_bebop2_algorithm(args):
    if args.policy_type == "bc_ppo":
        policy_kwargs = dict(
            bc_config_path=args.bc_config,
            bc_checkpoint_path=args.bc_checkpoint,
            project_root=PROJECT_ROOT,
            value_hidden_dim=args.bc_value_hidden_dim,
            max_log_std=args.max_log_std,
            log_std_init=args.log_std_init,
        )
        return PPO, BCInitializedActorCriticPolicy, policy_kwargs
    if args.policy_type == "residual_ppo":
        ppo_args = argparse.Namespace(**vars(args))
        ppo_args.policy_type = "ppo"
        return resolve_algorithm(ppo_args)
    return resolve_algorithm(args)


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.render:
        render(parsed_args)
    else:
        train(parsed_args)
