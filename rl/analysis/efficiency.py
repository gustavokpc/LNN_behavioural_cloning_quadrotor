"""Policy-only inference benchmarking."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .io import write_json, write_rows_csv
from .loading import load_model_and_env, model_information


def benchmark_inference(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    architecture: str = "auto",
    device: str = "auto",
    seed: int = 0,
    warmup: int = 100,
    iterations: int = 1000,
    env_overrides: dict | None = None,
) -> dict[str, object]:
    """Benchmark batch-one policy inference, excluding environment and load time."""

    import torch

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive.")
    model, env, args, _ = load_model_and_env(
        checkpoint,
        architecture=architecture,
        device=device,
        seed=seed,
        env_overrides=env_overrides,
    )
    observation = env.reset()
    env.close()
    recurrent_state = None
    episode_start = np.ones((1,), dtype=bool)

    def synchronize() -> None:
        if str(model.device).startswith("cuda"):
            torch.cuda.synchronize(model.device)

    with torch.inference_mode():
        for _ in range(warmup):
            _, recurrent_state = model.predict(
                observation, state=recurrent_state, episode_start=episode_start, deterministic=True
            )
            episode_start[:] = False
        synchronize()
        latencies = np.empty(iterations, dtype=np.float64)
        for index in range(iterations):
            synchronize()
            start = time.perf_counter_ns()
            _, recurrent_state = model.predict(
                observation, state=recurrent_state, episode_start=episode_start, deterministic=True
            )
            synchronize()
            latencies[index] = (time.perf_counter_ns() - start) * 1e-9
    mean = float(np.mean(latencies))
    result = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "architecture": args.policy_type,
        "device": str(model.device),
        "batch_size": 1,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "mean_latency_s": mean,
        "median_latency_s": float(np.median(latencies)),
        "p95_latency_s": float(np.percentile(latencies, 95)),
        "p99_latency_s": float(np.percentile(latencies, 99)),
        "std_latency_s": float(np.std(latencies, ddof=1)) if iterations > 1 else 0.0,
        "theoretical_frequency_hz": float(1.0 / mean),
        "nominal_control_frequency_hz": float(1.0 / args.dt),
        "real_time_factor": float(args.dt / mean),
        **model_information(model, checkpoint),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "latencies.npz", latency_s=latencies)
    write_json(output / "efficiency.json", result)
    write_rows_csv(output / "efficiency.csv", [result])
    return result
