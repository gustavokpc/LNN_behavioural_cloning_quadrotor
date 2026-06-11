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
from .envs.bebop2_waypoints_env import DEFAULT_SQUARE_WAYPOINTS, Bebop2WaypointEnv, ResidualBebop2WaypointEnv
from .legacy_ppo.drone_ppo_sb3 import (
    GradientEpisodePrintCallback,
    attach_gradient_logger,
    resolve_algorithm,
)
from ..utils.animation import animate
from ..utils.quadrotor_sim import body_to_world_state


RL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RL_ROOT.parent


def _state_to_absolute_position(state: np.ndarray, target: np.ndarray) -> np.ndarray:
    return target + body_to_world_state(state)[0:3]


def make_env(args: argparse.Namespace, num_envs: int, seed: int | None = None):
    env_kwargs = dict(
        num_envs=num_envs,
        waypoint_radius=args.waypoint_radius,
        dt=args.dt,
        max_steps=args.max_steps,
        integration_method=args.integration_method,
        implicit_iters=args.implicit_iters,
        initialize_at_random_waypoints=args.initialize_at_random_waypoints,
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
    ckpt_dir = RL_ROOT / "checkpoints" / "bebop2_waypoints" / algo_tag
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
        name_prefix=f"{algo_tag}_bebop2_waypoints",
    )
    attach_gradient_logger(model.policy)
    grad_cb = GradientEpisodePrintCallback()

    start = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, grad_cb],
        progress_bar=True,
    )
    model.save(ckpt_dir / f"{algo_tag}_bebop2_waypoints")
    env.close()
    elapsed = (time.time() - start) / 3600
    print(f"Finished training {args.total_timesteps:,} steps in {elapsed:0.2f}h. Latest checkpoint saved to {ckpt_dir}.")


def _collect_render_episode(model, args: argparse.Namespace):
    env = make_env(args, num_envs=1, seed=args.seed)
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
        episode_start = done
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
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-log-std", type=float, default=1.0)
    parser.add_argument("--rollout-fragment-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--vf-coeff", type=float, default=0.5)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--tensorboard-log", type=str, default=str(RL_ROOT / "runs" / "bebop2_waypoints"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-epochs", type=int, default=10)
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
    parser.add_argument("--waypoint-radius", type=float, default=0.2)
    parser.add_argument("--integration-method", default="rk4")
    parser.add_argument("--implicit-iters", type=int, default=1)
    parser.add_argument("--initialize-at-random-waypoints", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-steps", type=int, default=2000)
    parser.add_argument("--render-episodes", type=int, default=1)
    parser.add_argument("--simultaneous", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", default=str(RL_ROOT / "runs" / "bebop2_waypoints_rollout.mp4"))
    parser.add_argument("--auto-play", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_bebop2_algorithm(args):
    if args.policy_type == "bc_ppo":
        policy_kwargs = dict(
            bc_config_path=args.bc_config,
            bc_checkpoint_path=args.bc_checkpoint,
            project_root=PROJECT_ROOT,
            value_hidden_dim=args.bc_value_hidden_dim,
            max_log_std=args.max_log_std,
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
