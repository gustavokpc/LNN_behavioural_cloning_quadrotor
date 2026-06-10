# C Controller Exports

Each subfolder contains a C export for one checkpoint:

```text
MLP
LTC
RNN
CONV_CFC_DEFAULT
CFC
CFC_PURE
CTRNN
GRU
LSTM
NCP_CFC
```

Common files inside each folder:

```text
nn_parameters.h/.c   exported weights, biases, normalization limits
nn_operations.h/.c   inference implementation
run_controller.c     command-line runner that receives the input vector
test_controller.c    fixed-input C smoke test
compare_python_c.py  PyTorch-vs-C numerical comparison
README.md            model-specific notes
```

For recurrent/liquid models, `nn_control` keeps hidden state in static storage. Call `nn_reset()` before starting a new trajectory or whenever the controller state should be cleared.

Run a comparison from the repository root:

```sh
.venv/bin/python LNN_behavioural_cloning_quadrotor/C_codes/GRU/compare_python_c.py
```

The generated comparisons were checked against PyTorch with max absolute error below `6e-7` for the included test input.

C-backed simulators live in `simulators/`:

```sh
python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_random_start_C
python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_start_dataset_C
python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_race_drone_C
```

By default they select the C export from `simulator_config.yaml -> model_path`. To force a specific C export folder:

```sh
python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_random_start_C \
  --c-model-dir LNN_behavioural_cloning_quadrotor/C_codes/GRU
```
