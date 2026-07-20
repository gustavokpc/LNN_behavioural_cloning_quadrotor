from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT.parent
sys.path.insert(0, str(REPO_ROOT))

from LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 import make_env, resolve_bebop2_algorithm

C_DIR = Path(__file__).resolve().parent
CKPT = PROJECT / "rl/checkpoints/figure8_gates/recurrent_ppo/recurrent_ppo_figure8_gates.zip"
POLICY_TYPE = "recurrent_ppo"


def compile_lib() -> Path:
    lib = C_DIR / "libcontroller.so"
    subprocess.run(["gcc", "-std=c99", "-Wall", "-Wextra", "-pedantic", "-fPIC", "-shared", str(C_DIR / "nn_cfc_operations.c"), str(C_DIR / "nn_cfc_parameters.c"), "-lm", "-o", str(lib)], check=True)
    return lib


def load_policy():
    args = argparse.Namespace(policy_type=POLICY_TYPE, track="figure8_gates", num_envs=1, seed=0, gates_ahead=1, gate_size=1.5, dt=0.01, max_steps=6000, integration_method="rk4", implicit_iters=1, initialize_at_random_gates=False, initialize_uniform=False, num_state_history=0, num_action_history=0, history_step_size=1, param_input=False, param_input_noise=0.0, low_obs=False, no_vel=False, no_ang_vel=False, randomize_external_moments=True, normalize_observations=True, device="cpu", use_flatten_features=True, max_log_std=-1.5, cell_size=64, cfc_timespan=0.01, bc_config="", bc_checkpoint="", residual_scale=0.05)
    algo_cls, policy_class, _ = resolve_bebop2_algorithm(args)
    env = make_env(args, num_envs=1, seed=123)
    model = algo_cls.load(CKPT, env=env, device="cpu", custom_objects={"policy_class": policy_class})
    return model, env


def main() -> None:
    lib_path = compile_lib()
    lib = ctypes.CDLL(str(lib_path))
    arr_in = ctypes.c_float * 20
    arr4 = ctypes.c_float * 4
    lib.nn_cfc_reset.argtypes = []
    lib.nn_cfc_control.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]

    model, env = load_policy()
    obs = env.reset()
    py_state = None
    py_episode_start = np.array([True], dtype=bool)
    lib.nn_cfc_reset()
    max_err = 0.0
    for idx in range(20):
        py_action, py_state = model.predict(obs, state=py_state, episode_start=py_episode_start, deterministic=True)
        py_episode_start[:] = False
        c_out = arr4()
        lib.nn_cfc_control(arr_in(*obs.reshape(-1).astype(np.float32).tolist()), c_out)
        c_action = np.array(list(c_out), dtype=np.float32)
        py_motor = np.clip((py_action.reshape(-1) + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
        err = float(np.max(np.abs(py_motor - c_action)))
        max_err = max(max_err, err)
        print(f"step={idx:02d} py_motor={py_motor} c={c_action} max_abs_err={err:.9g}")
        obs, _, done, _ = env.step(py_action)
        if done.any():
            py_episode_start[:] = True
            py_state = None
            lib.nn_cfc_reset()
    env.close()
    print(f"max_abs_err={max_err:.9g}")
    if max_err > 2e-5:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
