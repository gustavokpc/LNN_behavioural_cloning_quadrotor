# RL CfC tau 0.025 noise — 96,000,000 steps

Paparazzi-compatible deterministic actor export from:

`rl/checkpoints/CFC/cfc_tau_0025/recurrent_ppo_figure8_gates_cfc_tau_0025_noise_96000000_steps.zip`

## Files to replace later

Copy only these four files into
`paparazzi/sw/airborne/modules/rl_cfc_control/`:

- `rl_cfc_operations.c`
- `rl_cfc_operations.h`
- `rl_cfc_parameters.c`
- `rl_cfc_parameters.h`

Keep the existing Paparazzi `rl_cfc_control.c` and `rl_cfc_control.h`
unchanged. Generating and validating this export does not modify the Paparazzi
checkout.

## Checkpoint contract

- training step: `96,000,000`;
- observation dimension: 20;
- observation order expected by the existing wrapper:
  `pos_G[3], vel_G[3], euler_B_to_G[3], rates_B[3], motor_state[4], next_gate_G[4]`;
- recurrent hidden-state dimension: 64;
- deterministic action dimension: 4;
- trained action range: `[0, 1]`;
- controller cadence: 100 Hz;
- CfC recurrent timespan: `0.01 s`.

The `0.025 s` value in the checkpoint name is the simulated motor time
constant used for this training run. It is not the CfC recurrent timespan and
therefore is not embedded in `rl_cfc_parameters.h`. Motor tau, physical rotor
yaw sign, and optional observation-noise injection are simulation/airframe
settings outside these four inference files.

To reproduce the exact test conditions used with this model, the Paparazzi
NPS/Gazebo airframe must separately use:

- four motor time constants of `0.025 s`;
- physical rotor-yaw sign `+1`;
- p/q/r observation-noise standard deviations `0.0, 0.0, 0.0`.

At export time, the installed `bebop2_cfc_sim.xml` still used motor tau `0.02 s`
and nonzero p/q/r observation noise. Those files were intentionally not changed.

## Regenerate and validate

From the workspace root:

```sh
.venv/bin/python LNN_behavioural_cloning_quadrotor/C_codes/bebop2/recurrent_ppo_figure8_gates_cfc_tau_0025_noise_96000000_steps/export_rl_cfc.py
.venv/bin/python LNN_behavioural_cloning_quadrotor/C_codes/bebop2/recurrent_ppo_figure8_gates_cfc_tau_0025_noise_96000000_steps/validate_export.py
gcc -std=c99 -Wall -Wextra -Werror -pedantic -fsyntax-only \
  LNN_behavioural_cloning_quadrotor/C_codes/bebop2/recurrent_ppo_figure8_gates_cfc_tau_0025_noise_96000000_steps/rl_cfc_operations.c \
  LNN_behavioural_cloning_quadrotor/C_codes/bebop2/recurrent_ppo_figure8_gates_cfc_tau_0025_noise_96000000_steps/rl_cfc_parameters.c
```
