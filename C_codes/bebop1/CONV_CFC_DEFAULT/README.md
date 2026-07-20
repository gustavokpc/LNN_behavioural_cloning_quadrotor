# CONV_CFC_DEFAULT C Export

Checkpoint: `conv_cfc_default_n64_epoch=17_val_loss=0.000326.ckpt`

Model kind: `cfc_default`

Input order:

```text
dx, dy, dz, vx, vy, vz, phi, theta, psi, p, q, r, Mx_ext, My_ext, Mz_ext, omega1, omega2, omega3, omega4, dt
```

Build and run fixed test:

```sh
gcc -std=c99 -Wall -Wextra test_controller.c nn_operations.c nn_parameters.c -lm -o test_controller
./test_controller
```

Compare against PyTorch:

```sh
../../../.venv/bin/python compare_python_c.py
```

For recurrent models, `nn_control` keeps hidden state in static storage. Call `nn_reset()` before a new trajectory.
