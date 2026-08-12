# Bebop2 delayed correct-sign Conv-CfC

This directory is a Paparazzi-ready C export of:

`checkpoints/bebop2/conv_cfc_default_n64_bebop2_delayed_correct_sign.ckpt`

The four generated `nn_cfc_operations.*` and `nn_cfc_parameters.*` files match
the existing Paparazzi controller interface in
`sw/airborne/modules/nn_cfc_control/`. Keep the existing
`nn_cfc_control.c/.h` unchanged. No file in the Paparazzi checkout is modified
by this export.

## Important model semantics

- Physical input order:
  `dx, dy, dz, vx, vy, vz, phi, theta, psi, p, q, r, Mx_ext, My_ext, Mz_ext, omega1, omega2, omega3, omega4`.
- Inputs are normalized internally with the `bebop2_tau_0_06` limits.
- Outputs are clipped to `[0, 1]` and mapped by the wrapper to `5000..10000 RPM`.
- The checkpoint was trained without `dt`. Its CfC cell must use the ncps
  implicit timespan `1.0`; the wrapper's measured 100 Hz period is intentionally
  not passed into the network.
- Keep the Paparazzi module scheduled at the training cadence of `100 Hz`
  (`freq="100."` in `conf/modules/nn_cfc_control.xml`).
- The one-step delay was applied when aligning training inputs and targets. It is
  already learned by the controller and must not be added again at runtime.
- The existing Paparazzi wrapper continues to own coordinate conversion, motor
  order, RPM feedback, safety gating, targets, telemetry, and logging.

## Plug into Paparazzi

After making a backup or committing the Paparazzi working tree, copy only these
four files:

`nn_cfc_operations.c`, `nn_cfc_operations.h`, `nn_cfc_parameters.c`,
`nn_cfc_parameters.h`

into:

`paparazzi/sw/airborne/modules/nn_cfc_control/`

The filenames are already the ones expected by
`conf/modules/nn_cfc_control.xml`. The existing
`bebop2_nn_cfc_control` airframe can then be compiled normally.

## Reproduce and validate

From this directory:

```sh
../../../../.venv/bin/python export_conv_cfc.py
../../../../.venv/bin/python compare_python_c.py
gcc -std=c99 -Wall -Wextra -Werror -pedantic \
  -c nn_cfc_operations.c nn_cfc_parameters.c
```

`compare_python_c.py` executes the same recurrent sequence in PyTorch and in
the generated C implementation and fails if they differ beyond the configured
floating-point tolerance.
