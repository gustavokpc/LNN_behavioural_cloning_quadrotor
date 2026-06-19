# RECURRENT_PPO_57600000

Checkpoint: `recurrent_ppo_bebop2_waypoints_57600000_steps.zip`

This export implements the deterministic actor from the SB3 RecurrentPPO CfC policy:

- raw 19-state input
- internal min/max observation normalization
- CfC actor recurrent hidden state with 64 units
- policy MLP and clipped 4-motor action output in `[0, 1]`

Build and smoke-test:

```bash
gcc -std=c99 -Wall -Wextra test_controller.c nn_operations.c nn_parameters.c -lm -o test_controller
./test_controller
```

Compare against PyTorch/SB3 from the repository root:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/C_codes/RECURRENT_PPO_57600000/compare_python_c.py
```

For trajectory simulations call `nn_reset()` at the start of each rollout. `nn_control()` keeps the recurrent hidden state between calls.
