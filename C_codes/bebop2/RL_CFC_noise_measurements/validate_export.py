from __future__ import annotations

import ctypes
import io
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
CHECKPOINT = (
    PROJECT
    / "rl/checkpoints/CFC/recurrent_ppo_with_noise_measurements/"
    "recurrent_ppo_figure8_gates_noise_51200000_steps.zip"
)


def load_state_dict() -> dict[str, torch.Tensor]:
    with zipfile.ZipFile(CHECKPOINT) as archive:
        return torch.load(
            io.BytesIO(archive.read("policy.pth")),
            map_location="cpu",
            weights_only=False,
        )


def linear(x: torch.Tensor, state: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return torch.nn.functional.linear(x, state[prefix + ".weight"], state[prefix + ".bias"])


def python_actor_step(
    observation: torch.Tensor,
    hidden: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.cat((observation, hidden))
    x = 1.7159 * torch.tanh(0.666 * linear(x, state, "lstm_actor.rnn_cell.backbone.0"))
    ff1 = torch.tanh(linear(x, state, "lstm_actor.rnn_cell.ff1"))
    ff2 = torch.tanh(linear(x, state, "lstm_actor.rnn_cell.ff2"))
    time_a = linear(x, state, "lstm_actor.rnn_cell.time_a")
    time_b = linear(x, state, "lstm_actor.rnn_cell.time_b")
    interp = torch.sigmoid(time_a * 0.01 + time_b)
    hidden = ff1 * (1.0 - interp) + interp * ff2
    policy = torch.tanh(linear(hidden, state, "mlp_extractor.policy_net.0"))
    policy = torch.tanh(linear(policy, state, "mlp_extractor.policy_net.2"))
    raw = linear(policy, state, "action_net")
    action01 = torch.clamp(raw, 0.0, 1.0)
    return action01, hidden, raw


def main() -> None:
    library = Path("/tmp/librl_cfc_noise_measurements_51200000.so")
    subprocess.run(
        [
            "gcc", "-std=c99", "-O3", "-Wall", "-Wextra", "-Werror",
            "-fPIC", "-shared",
            str(HERE / "rl_cfc_operations.c"),
            str(HERE / "rl_cfc_parameters.c"),
            "-lm", "-o", str(library),
        ],
        check=True,
    )

    lib = ctypes.CDLL(str(library))
    float_ptr = ctypes.POINTER(ctypes.c_float)
    lib.rl_cfc_reset.argtypes = []
    lib.rl_cfc_control.argtypes = [float_ptr, float_ptr]
    lib.rl_cfc_get_hidden.argtypes = [float_ptr]
    raw_c = (ctypes.c_float * 4).in_dll(lib, "rl_cfc_last_raw_control")

    state = load_state_dict()
    rng = np.random.default_rng(51200000)
    observations = rng.normal(0.0, 1.0, size=(500, 20)).astype(np.float32)
    observations[:, 12:16] = rng.uniform(-1.0, 1.0, size=(500, 4))

    hidden_py = torch.zeros(64, dtype=torch.float32)
    lib.rl_cfc_reset()
    max_action_error = 0.0
    max_hidden_error = 0.0
    max_raw_error = 0.0

    for observation in observations:
        action_py, hidden_py, raw_py = python_actor_step(
            torch.from_numpy(observation), hidden_py, state
        )
        input_c = (ctypes.c_float * 20)(*observation)
        output_c = (ctypes.c_float * 4)()
        hidden_c = (ctypes.c_float * 64)()
        lib.rl_cfc_control(input_c, output_c)
        lib.rl_cfc_get_hidden(hidden_c)

        action_c = np.asarray(output_c, dtype=np.float32)
        max_action_error = max(max_action_error, float(np.max(np.abs(action_c - action_py.numpy()))))
        max_hidden_error = max(max_hidden_error, float(np.max(np.abs(np.asarray(hidden_c) - hidden_py.numpy()))))
        max_raw_error = max(max_raw_error, float(np.max(np.abs(np.asarray(raw_c) - raw_py.numpy()))))

    print(f"steps={len(observations)}")
    print(f"max_action_abs_error={max_action_error:.9g}")
    print(f"max_hidden_abs_error={max_hidden_error:.9g}")
    print(f"max_raw_abs_error={max_raw_error:.9g}")
    if max(max_action_error, max_hidden_error, max_raw_error) > 2.0e-5:
        raise SystemExit("C export validation failed")


if __name__ == "__main__":
    main()
