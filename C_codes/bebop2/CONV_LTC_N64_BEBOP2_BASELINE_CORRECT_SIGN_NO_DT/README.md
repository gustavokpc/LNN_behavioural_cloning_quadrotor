# Conv-LTC N64 Bebop2 corrected sign — no dt

Paparazzi-compatible deterministic controller export from:

- checkpoint: `checkpoints/last_ones/conv_ltc_n64_bebop2_baseline_correct_sign_no_dt.ckpt`;
- config: `configs/last_ones/conv_ltc_n64_bebop2_baseline_correct_sign_no_dt.yaml`;
- normalization: `bebop2_tau_0_06`.

The existing `nn_cfc_control.c`, `nn_cfc_control.h`, and moment-observer files
do not need to change. When ready, replace only:

- `nn_cfc_operations.c`;
- `nn_cfc_operations.h`;
- `nn_cfc_parameters.c`;
- `nn_cfc_parameters.h`.

This model has 19 physical inputs, Conv1D preprocessing, a fully connected
64-state LTC with 6 ODE unfolds, and four normalized motor outputs. Because it
was trained without `dt`, its LTC step reproduces ncps' implicit
`elapsed_time=1.0`. The Paparazzi control loop still runs at 100 Hz; `1.0` is
the model's internal recurrent timespan, not the physical controller period.

The export and validation do not modify the Paparazzi checkout.
