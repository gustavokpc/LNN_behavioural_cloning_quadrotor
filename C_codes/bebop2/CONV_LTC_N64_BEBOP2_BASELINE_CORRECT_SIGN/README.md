# Conv-LTC N64 Bebop2 corrected sign — runtime dt

Paparazzi-compatible deterministic controller export from:

- checkpoint: `checkpoints/last_ones/conv_ltc_n64_bebop2_baseline_correct_sign.ckpt`;
- config: `configs/last_ones/conv_ltc_n64_bebop2_baseline_correct_sign.yaml`;
- normalization: `bebop2_tau_0_06`.

The existing `nn_cfc_control.c`, `nn_cfc_control.h`, and moment-observer files
do not need to change. When ready, replace only:

- `nn_cfc_operations.c`;
- `nn_cfc_operations.h`;
- `nn_cfc_parameters.c`;
- `nn_cfc_parameters.h`.

This model has 19 physical inputs, Conv1D preprocessing, a fully connected
64-state LTC with 6 ODE unfolds, and four normalized motor outputs. Its
`nn_cfc_operations.h` enables the runtime-timespan hook already supported by
the existing control wrapper, so each inference receives the measured control
period. The fallback/default period is `0.01 s`.

The export and validation do not modify the Paparazzi checkout.
