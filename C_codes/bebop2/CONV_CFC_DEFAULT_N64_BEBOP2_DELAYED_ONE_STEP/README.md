# Conv-CfC default n64 Bebop2 delayed one step

Exact C export of:

`checkpoints/bebop2/conv_cfc_default_n64_bebop2_delayed_one_step.ckpt`

The network consumes 19 normalized physical state values. The `dt` channel
used during training is not part of that vector: it is passed separately to
the CfC cell through `nn_set_timespan()`.

The included Paparazzi wrapper converts `nn_cfc_control_periodic_dt_us` from
microseconds to seconds before every inference. The first call after a reset
uses the nominal training period of `0.01 s`.

Validation performed:

- strict C99 compilation with warnings treated as errors for the exported
  operations and parameters;
- Python/C recurrent comparison over 100 steps with varying `dt`;
- Paparazzi NPS build for `bebop2_nn_cfc`;
- Paparazzi embedded `ap` cross-build for `bebop2_nn_cfc`.
