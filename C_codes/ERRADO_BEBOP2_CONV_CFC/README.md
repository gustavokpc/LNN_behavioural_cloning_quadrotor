# BBP2_CONV_CFC C Export

Checkpoint: `bbp2_conv_cfc_default_n64_epoch=19_val_loss=0.000143.ckpt`

Model kind: `cfc_default` with convolutional front-end and ReLU backbone activation.

The original checkpoint was trained with `dt` in the Python input labels, but
the Lightning wrapper removes `dt` from the neural-network feature vector and
passes it separately as CfC `timespans`. For embedded use, this export absorbs
the `0.01` CfC timespan into the exported `time_a` weights and biases.
`nn_control()` receives only the 19 physical state inputs; no `dt` or
timespan is passed at runtime.

Input normalization uses `utils/normalization_limits_bebop2.py`.

Input order:

```text
dx, dy, dz, vx, vy, vz, phi, theta, psi, p, q, r, Mx_ext, My_ext, Mz_ext, omega1, omega2, omega3, omega4
```

Build and run fixed test:

```sh
gcc -std=c99 -Wall -Wextra test_controller.c nn_operations.c nn_parameters.c -lm -o test_controller
./test_controller
```

Use in square simulator:

```sh
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_gazebo_square_C \
  --model-config LNN_behavioural_cloning_quadrotor/configs/bbp2_conv_cfc_default_n64_epoch=19_val_loss=0.000143.yaml \
  --c-model-dir LNN_behavioural_cloning_quadrotor/C_codes/BBP2_CONV_CFC \
  --dynamics-model quadrotor_sim_matlab \
  --time-simulation 20 \
  --dist-error 0.1
```
