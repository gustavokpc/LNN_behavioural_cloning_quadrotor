# RL CFC noise, motor tau 0.04 s

Paparazzi-compatible C export of the deterministic actor from:

`rl/checkpoints/CFC/rl_models/recurrent_ppo_figure8_gates_ckpt_96000000_steps.zip`

The `rl_cfc_control.*` and `rl_cfc_operations.*` files follow the currently
installed Paparazzi module exactly. Only the checkpoint-identification comment
and the generated actor parameters differ.

All simulation-only configuration is owned by the Paparazzi NPS/Gazebo
airframe `bebop2_cfc_sim.xml`. It is the single source for the `0.04 s`
per-motor time constants, physical rotor-yaw sign, and RL observation-rate
noise parameters.
The C wrapper consumes those generated `NPS_*` definitions and contains no
fallback simulation values. Policy observations use the normal yaw sign; the
physical rotor-yaw inversion is applied only in the Gazebo dynamics.

Regenerate the parameters:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/C_codes/bebop2/RL_CFC_noise_tau_004/export_rl_cfc.py
```

Validate the C actor against PyTorch for 500 recurrent steps:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/C_codes/bebop2/RL_CFC_noise_tau_004/validate_export.py
```

The matching controller is installed in the Paparazzi tree.
