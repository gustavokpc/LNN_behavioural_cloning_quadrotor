# Conv-CfC default n64 Bebop2 no-dt delayed

Exact C export of:

`checkpoints/bebop2/conv_cfc_default_n64_bebop2_no_dt_delayed.ckpt`

The network consumes 19 physical state values in this order:

`dx, dy, dz, vx, vy, vz, phi, theta, psi, p, q, r, Mx_ext, My_ext, Mz_ext, omega1, omega2, omega3, omega4`

The inputs are normalized internally with the `bebop2_tau_0_06` profile. The
four outputs are clamped to `[0, 1]` and the Paparazzi wrapper maps them to
motor RPM.

This checkpoint was trained without `dt`. Its CfC cell therefore uses the
library's implicit timespan of exactly `1.0`; the C API intentionally has no
`nn_set_timespan()` function and the Paparazzi wrapper never passes the
measured controller period into the network.

The six `nn_cfc_*.[ch]` files can replace the files in Paparazzi's
`sw/airborne/modules/nn_cfc_control/` directory. This export directory does
not modify the Paparazzi checkout.

Validation:

- regenerate parameters: `python export_conv_cfc.py`
- compare 100 recurrent steps against PyTorch: `python compare_python_c.py`
- strict C99 build: `gcc -std=c99 -Wall -Wextra -Werror -pedantic -c nn_cfc_operations.c nn_cfc_parameters.c`
