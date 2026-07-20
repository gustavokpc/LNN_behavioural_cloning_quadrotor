# FIGURE8_GATES_LTC_RECURRENT_PPO

Checkpoint: `recurrent_ppo_figure8_gates.zip`

This export implements the deterministic actor from the SB3 Figure-8 Gates RecurrentPPO LTC policy.

- raw Figure-8 Gates RL observation input (`NUM_STATES=20`)
- recurrent actor hidden state with 64 units
- action head raw output in legacy `[-1, 1]` stored in `nn_cfc_last_raw_control`
- public `nn_cfc_control()` output converted to motor command `[0, 1]` for the plant/Paparazzi

Build and smoke-test:

```bash
gcc -std=c99 -Wall -Wextra test_controller.c nn_cfc_operations.c nn_cfc_parameters.c -lm -o test_controller
./test_controller
```

Compare against PyTorch/SB3 from the repository root:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/C_codes/FIGURE8_GATES_LTC_RECURRENT_PPO/compare_python_c.py
```

Call `nn_cfc_reset()` at the start of each rollout. `nn_cfc_control()` keeps recurrent hidden state between calls.
