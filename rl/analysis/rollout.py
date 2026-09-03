"""Central deterministic rollout collector used by every evaluation mode."""

from __future__ import annotations

import dataclasses
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .environment import (
    apply_initial_condition_scale,
    apply_state_perturbation,
    command_coordinates,
    install_terminal_state_capture,
    true_state_snapshot,
)
from .io import summary_rows, write_json, write_rows_csv
from .loading import load_model_and_env, model_information, recover_checkpoint_step
from .metrics import aggregate_episodes, compute_episode_metrics


@dataclasses.dataclass
class DropoutConfig:
    kind: str = "none"  # none, velocity, angular_rate, full
    mode: str = "zero"  # zero, freeze
    onset_s: float = 1.0
    duration_s: float = 0.2
    onset_event: str = "time"  # time, first_gate_crossing, perturbation_onset


@dataclasses.dataclass
class InitialConditionConfig:
    position_scale: float = 1.0
    velocity_scale: float = 1.0
    attitude_scale: float = 1.0
    angular_rate_scale: float = 1.0


@dataclasses.dataclass
class PerturbationConfig:
    onset_s: float = -1.0
    values: dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RolloutConfig:
    checkpoint: str
    output_dir: str
    architecture: str = "auto"
    label: str = "evaluation"
    episodes: int = 10
    seed: int = 0
    device: str = "auto"
    record_recurrent: bool = False
    record_internals: bool = False
    env_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    policy_dt: float | None = None
    dropout: DropoutConfig = dataclasses.field(default_factory=DropoutConfig)
    initial_condition: InitialConditionConfig = dataclasses.field(default_factory=InitialConditionConfig)
    perturbation: PerturbationConfig = dataclasses.field(default_factory=PerturbationConfig)


def _hidden_array(state) -> np.ndarray | None:
    if state is None:
        return None
    hidden = state[0] if isinstance(state, (tuple, list)) else state
    hidden = np.asarray(hidden)
    if hidden.size == 0:
        return None
    # SB3 layout is (layers, environments, hidden); evaluator uses one env.
    return hidden.reshape(-1).astype(np.float32)


def _incoming_hidden_tensor(policy, recurrent_state, hidden_size: int, reference):
    import torch

    hidden = _hidden_array(recurrent_state)
    if hidden is None:
        return torch.zeros((1, hidden_size), dtype=reference.dtype, device=reference.device)
    return torch.as_tensor(hidden, dtype=reference.dtype, device=reference.device).reshape(1, hidden_size)


def exact_recurrent_internal(model, observation: np.ndarray, recurrent_state) -> tuple[str | None, np.ndarray | None, str]:
    """Extract an implementation-exact CfC gate or LTC effective time constant.

    CfC uses ``sigmoid(time_a(backbone([x,h])) * dt + time_b(...))``.
    LTC uses ``cm / (gleak + sum recurrent conductance + sum sensory
    conductance)`` at the incoming state.  Wired CfC is deliberately unsupported
    because it has multiple layer-local gates rather than one policy-state gate.
    """

    import torch

    policy = model.policy
    recurrent = getattr(policy, "lstm_actor", None)
    cell = getattr(recurrent, "rnn_cell", None)
    if recurrent is None or cell is None:
        return None, None, "unsupported: policy has no ncps recurrent cell"
    name = cell.__class__.__name__
    with torch.inference_mode():
        obs_tensor, _ = policy.obs_to_tensor(np.asarray(observation, dtype=np.float32).reshape(1, -1))
        features = policy.extract_features(obs_tensor)
        if isinstance(features, tuple):
            features = features[0]
        hidden_size = int(getattr(recurrent, "state_size", getattr(recurrent, "hidden_size", 0)))
        hidden = _incoming_hidden_tensor(policy, recurrent_state, hidden_size, features)
        timespan = getattr(recurrent, "cfc_timespan", None)
        dt = float(timespan if timespan is not None else 1.0)
        if name == "CfCCell":
            combined = torch.cat([features, hidden], dim=1)
            if cell.backbone_layers > 0:
                combined = cell.backbone(combined)
            if cell.mode == "pure":
                rate = torch.abs(cell.w_tau) + torch.abs(cell.ff1(combined))
                tau = 1.0 / torch.clamp(rate, min=1e-12)
                return "cfc_tau", tau[0].detach().cpu().numpy().astype(np.float32), (
                    "exact pure-CfC tau=1/(|w_tau|+|ff1(backbone([x,h]))|)"
                )
            gate = torch.sigmoid(cell.time_a(combined) * dt + cell.time_b(combined))
            return "cfc_gate", gate[0].detach().cpu().numpy().astype(np.float32), (
                "exact CfC interpolation gate sigmoid(time_a(backbone([x,h]))*dt+time_b(backbone([x,h])))"
            )
        if name == "WiredCfCCell":
            return None, None, "unsupported: wired NCP CfC has layer-local states/gates not exposed as one aligned vector"
        if name == "LTCCell":
            inputs = cell._map_inputs(features)
            positive = cell.make_positive_fn
            sensory = positive(cell._params["sensory_w"]) * cell._sigmoid(
                inputs, cell._params["sensory_mu"], cell._params["sensory_sigma"]
            )
            sensory = sensory * cell._params["sensory_sparsity_mask"]
            recurrent_g = positive(cell._params["w"]) * cell._sigmoid(
                hidden, cell._params["mu"], cell._params["sigma"]
            )
            recurrent_g = recurrent_g * cell._params["sparsity_mask"]
            denominator = positive(cell._params["gleak"]) + sensory.sum(dim=1) + recurrent_g.sum(dim=1)
            tau = positive(cell._params["cm"]) / torch.clamp(denominator, min=cell._epsilon)
            return "ltc_tau", tau.detach().cpu().numpy().reshape(-1).astype(np.float32), (
                "exact instantaneous LTC tau=cm/(gleak+sum sensory conductance+sum recurrent conductance) at incoming (x,h)"
            )
    return None, None, f"unsupported recurrent cell: {name}"


def _dropout_indices(kind: str, observation_size: int) -> np.ndarray:
    if kind == "velocity":
        indices = np.arange(3, 6)
    elif kind == "angular_rate":
        indices = np.arange(9, 12)
    elif kind == "full":
        indices = np.arange(observation_size)
    elif kind == "none":
        indices = np.empty(0, dtype=int)
    else:
        raise ValueError(f"Unknown dropout kind: {kind}")
    if indices.size and indices[-1] >= observation_size:
        raise ValueError("Requested dropout is not available in this observation configuration.")
    return indices


def _corrupt_observation(
    observation: np.ndarray,
    step: int,
    dt: float,
    config: DropoutConfig,
    frozen: np.ndarray | None,
    onset_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    observation = np.asarray(observation, dtype=np.float32).copy()
    indices = _dropout_indices(config.kind, observation.shape[1])
    onset = int(round(config.onset_s / dt)) if onset_step is None else int(onset_step)
    if onset < 0:
        return observation, frozen, False
    duration = max(0, int(round(config.duration_s / dt)))
    active = bool(indices.size and onset <= step < onset + duration)
    if not active:
        return observation, frozen, False
    if frozen is None:
        frozen = observation.copy()
    if config.mode == "zero":
        observation[:, indices] = 0.0
    elif config.mode == "freeze":
        observation[:, indices] = frozen[:, indices]
    else:
        raise ValueError(f"Unknown dropout mode: {config.mode}")
    return observation, frozen, True


def _environment_metadata(args) -> dict[str, Any]:
    names = (
        "track", "dt", "max_steps", "integration_method", "implicit_iters",
        "gate_size", "gates_ahead", "figure8_action_range", "figure8_start_gate",
        "no_vel", "no_ang_vel", "low_obs", "normalize_observations", "num_state_history", "num_action_history",
        "history_step_size", "param_input", "param_input_noise", "obs_rate_noise_std",
        "motor_tau", "tau", "randomize_dynamics", "randomization_factor",
        "randomize_aerodynamic_coefficients", "randomize_external_moments",
    )
    return {name: getattr(args, name) for name in names if hasattr(args, name)}


def collect_rollouts(config: RolloutConfig) -> Path:
    """Run deterministic episodes and save raw/summary results."""

    if config.episodes <= 0:
        raise ValueError("episodes must be positive.")
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    model, env, args, training_metadata = load_model_and_env(
        config.checkpoint,
        architecture=config.architecture,
        device=config.device,
        seed=config.seed,
        env_overrides=config.env_overrides,
        policy_dt=config.policy_dt,
    )
    install_terminal_state_capture(env)
    all_steps: dict[str, list[np.ndarray | float | int | bool | str]] = {
        key: [] for key in (
            "episode_id", "step", "time_s", "true_state", "observation", "policy_observation",
            "policy_action", "motor_command_01", "motor_command_rpm", "actual_motor_speed",
            "reward", "target_index", "target_index_after", "gate_crossing", "terminated",
            "truncated", "termination_reason", "hidden_state", "dropout_active",
            "dropout_onset", "observation_restoration", "perturbation_onset",
        )
    }
    internal_values: list[np.ndarray] = []
    internal_name: str | None = None
    internal_formula = "not requested"
    episode_rows = []
    start_wall = time.perf_counter()
    try:
        for episode_id in range(config.episodes):
            episode_seed = config.seed + episode_id
            env.seed(episode_seed)
            observation = env.reset()
            apply_initial_condition_scale(
                env,
                position=config.initial_condition.position_scale,
                velocity=config.initial_condition.velocity_scale,
                attitude=config.initial_condition.attitude_scale,
                angular_rate=config.initial_condition.angular_rate_scale,
            )
            observation = env._observations().copy() if hasattr(env, "_observations") else env.states.copy()
            recurrent_state = None
            episode_start = np.ones((1,), dtype=bool)
            frozen_observation = None
            perturbation_applied = False
            ep_rewards: list[float] = []
            ep_actions: list[np.ndarray] = []
            ep_commands: list[np.ndarray] = []
            ep_rpm: list[np.ndarray] = []
            ep_true: list[np.ndarray] = []
            ep_gate: list[bool] = []
            final_info: Mapping[str, object] = {}
            done = False
            step = 0
            previous_dropout = False
            dropout_event_step: int | None = None
            while not done:
                perturb_onset = False
                if (
                    not perturbation_applied
                    and config.perturbation.values
                    and config.perturbation.onset_s >= 0.0
                    and step >= int(round(config.perturbation.onset_s / args.dt))
                ):
                    apply_state_perturbation(env, config.perturbation.values)
                    observation = env._observations().copy() if hasattr(env, "_observations") else env.states.copy()
                    perturbation_applied = True
                    perturb_onset = True
                    if config.dropout.onset_event == "perturbation_onset" and dropout_event_step is None:
                        dropout_event_step = step + int(round(config.dropout.onset_s / args.dt))
                raw_observation = np.asarray(observation, dtype=np.float32).copy()
                policy_observation, frozen_observation, dropout_active = _corrupt_observation(
                    raw_observation, step, args.dt, config.dropout, frozen_observation,
                    onset_step=(
                        None if config.dropout.onset_event == "time"
                        else (-1 if dropout_event_step is None else dropout_event_step)
                    ),
                )
                dropout_onset = dropout_active and not previous_dropout
                observation_restoration = previous_dropout and not dropout_active
                previous_dropout = dropout_active
                if config.record_internals:
                    name, values, formula = exact_recurrent_internal(model, policy_observation, recurrent_state)
                    internal_formula = formula
                    if name is not None and values is not None:
                        internal_name = name
                        internal_values.append(values)
                target_before = int(env.target_gates[0] if hasattr(env, "target_gates") else env.target_waypoints[0])
                action, next_recurrent_state = model.predict(
                    policy_observation,
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                command, rpm = command_coordinates(env, action[0])
                observation, reward, dones, infos = env.step(action)
                done = bool(dones[0])
                info = infos[0]
                state, target_after = true_state_snapshot(env, terminal=done)
                success_now = bool(info.get("is_success", False))
                from .metrics import termination_reasons

                reasons = termination_reasons(info, success=success_now) if done else []
                hidden = _hidden_array(next_recurrent_state) if config.record_recurrent else None
                all_steps["episode_id"].append(episode_id)
                all_steps["step"].append(step)
                all_steps["time_s"].append((step + 1) * args.dt)
                all_steps["true_state"].append(state.astype(np.float32))
                all_steps["observation"].append(raw_observation[0])
                all_steps["policy_observation"].append(policy_observation[0])
                all_steps["policy_action"].append(np.asarray(action[0], dtype=np.float32))
                all_steps["motor_command_01"].append(command.astype(np.float32))
                all_steps["motor_command_rpm"].append(rpm.astype(np.float32))
                all_steps["actual_motor_speed"].append(state[15:19].astype(np.float32))
                all_steps["reward"].append(float(reward[0]))
                all_steps["target_index"].append(target_before)
                all_steps["target_index_after"].append(target_after)
                all_steps["gate_crossing"].append(bool(info.get("gate_passed", info.get("waypoint_reached", False))))
                all_steps["terminated"].append(done and not bool(info.get("TimeLimit.truncated", False)))
                all_steps["truncated"].append(bool(info.get("TimeLimit.truncated", False)))
                all_steps["termination_reason"].append("+".join(reasons))
                all_steps["hidden_state"].append(hidden if hidden is not None else np.empty(0, dtype=np.float32))
                all_steps["dropout_active"].append(dropout_active)
                all_steps["dropout_onset"].append(dropout_onset)
                all_steps["observation_restoration"].append(observation_restoration)
                all_steps["perturbation_onset"].append(perturb_onset)
                ep_rewards.append(float(reward[0]))
                ep_actions.append(np.asarray(action[0], dtype=np.float64))
                ep_commands.append(command)
                ep_rpm.append(rpm)
                ep_true.append(state)
                ep_gate.append(bool(info.get("gate_passed", info.get("waypoint_reached", False))))
                if (
                    ep_gate[-1] and config.dropout.onset_event == "first_gate_crossing"
                    and dropout_event_step is None
                ):
                    dropout_event_step = step + 1 + int(round(config.dropout.onset_s / args.dt))
                recurrent_state = next_recurrent_state
                episode_start = dones
                final_info = info
                step += 1
            episode_metric = compute_episode_metrics(
                rewards=np.asarray(ep_rewards),
                actions=np.asarray(ep_actions),
                command_actions=np.asarray(ep_commands),
                rpm_commands=np.asarray(ep_rpm),
                true_states=np.asarray(ep_true),
                gate_events=np.asarray(ep_gate),
                dt=args.dt,
                final_info=final_info,
                num_gates=int(getattr(env, "num_gates", 1)),
            )
            episode_metric.update(episode_id=episode_id, evaluation_seed=episode_seed)
            episode_rows.append(episode_metric)
    finally:
        env.close()

    arrays: dict[str, np.ndarray] = {}
    for key, values in all_steps.items():
        if key == "hidden_state":
            nonempty = [value for value in values if np.asarray(value).size]
            width = np.asarray(nonempty[0]).size if nonempty else 0
            arrays[key] = np.asarray(values, dtype=np.float32).reshape(len(values), width) if width else np.empty((len(values), 0), dtype=np.float32)
        elif key == "termination_reason":
            arrays[key] = np.asarray(values, dtype="U64")
        else:
            arrays[key] = np.asarray(values)
    if internal_name and internal_values:
        arrays[internal_name] = np.asarray(internal_values, dtype=np.float32)
    np.savez_compressed(output / "steps.npz", **arrays)
    summary = aggregate_episodes(episode_rows)
    checkpoint = Path(config.checkpoint).resolve()
    metadata = {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint),
        "checkpoint_step": recover_checkpoint_step(checkpoint),
        "architecture": args.policy_type,
        "experiment_label": config.label,
        "training_seed": training_metadata.get("seed"),
        "evaluation_seed": config.seed,
        "number_of_episodes": config.episodes,
        "environment": _environment_metadata(args),
        "dt": args.dt,
        "policy_dt": config.policy_dt if config.policy_dt is not None else args.dt,
        "observation_configuration": {
            "no_velocity": bool(getattr(args, "no_vel", False)),
            "no_angular_rate": bool(getattr(args, "no_ang_vel", False)),
            "low_observation": bool(getattr(args, "low_obs", False)),
            "normalization": bool(
                getattr(args, "normalize_observations", False)
                and getattr(args, "track", "figure8_gates") != "figure8_gates"
            ),
        },
        "robustness_configuration": {
            "environment_overrides": config.env_overrides,
            "initial_condition": dataclasses.asdict(config.initial_condition),
            "dropout": dataclasses.asdict(config.dropout),
            "perturbation": dataclasses.asdict(config.perturbation),
            "dropout_stage": "after environment observation construction/normalization, immediately before policy",
        },
        "recurrent_recording": config.record_recurrent,
        "internal_recording": {
            "requested": config.record_internals,
            "quantity": internal_name,
            "formula": internal_formula,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_wall_s": time.perf_counter() - start_wall,
        "model": model_information(model, checkpoint),
        "training_metadata_path": training_metadata.get("_metadata_path"),
    }
    write_json(output / "metadata.json", metadata)
    write_json(output / "summary.json", summary)
    write_rows_csv(output / "episodes.csv", episode_rows)
    write_rows_csv(output / "summary.csv", summary_rows(summary))
    return output


def load_steps(result_dir: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(result_dir) / "steps.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
