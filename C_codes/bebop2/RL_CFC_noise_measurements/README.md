# RL CfC noise measurements — 51,200,000 steps

Paparazzi-compatible deterministic actor export from:

`rl/checkpoints/CFC/recurrent_ppo_with_noise_measurements/recurrent_ppo_figure8_gates_noise_51200000_steps.zip`

## Files to plug into Paparazzi

Copy only:

- `rl_cfc_operations.c`
- `rl_cfc_operations.h`
- `rl_cfc_parameters.c`
- `rl_cfc_parameters.h`

into `paparazzi/sw/airborne/modules/rl_cfc_control/`.

Keep the existing Paparazzi `rl_cfc_control.c` and `rl_cfc_control.h`
unchanged. This export does not create copies of those files and does not
modify the Paparazzi checkout.

## Checkpoint contract

- observation dimension: 20;
- observation order expected by the existing wrapper:
  `pos_G[3], vel_G[3], euler_B_to_G[3], rates_B[3], motor_state[4], next_gate_G[4]`;
- hidden-state dimension: 64;
- deterministic action dimension: 4;
- trained action range: `[0, 1]`;
- CfC timespan: `0.01 s`;
- controller cadence: `100 Hz`;
- checkpoint training step: `51,200,000`.

Measurement noise was a training-environment input augmentation. The exported
actor is deterministic and does not add random noise itself. On hardware, the
existing Paparazzi estimator supplies measured observations. Any optional
SITL-only noise injection remains owned by the existing wrapper/airframe
configuration.

## Reproduce and validate

From this directory:

```sh
../../../../.venv/bin/python export_rl_cfc.py
../../../../.venv/bin/python validate_export.py
gcc -std=c99 -Wall -Wextra -Werror -pedantic \
  -fsyntax-only rl_cfc_operations.c rl_cfc_parameters.c
```
