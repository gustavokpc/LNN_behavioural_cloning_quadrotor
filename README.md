# LNN Behavioural Cloning Quadrotor

This folder contains the supervised-learning pipeline for the quadrotor controller experiments. It is organized so the same saved configuration can be reused across:

- training
- checkpoint reconstruction
- offline testing
- automated feature ablations
- closed-loop simulation from datasets
- random-start robustness simulation
- race-track style gate simulation
- C export and C-backed simulation

## Layout

- [train.py](train.py)
  Main training entrypoint. Reads `train_config.yaml`, builds the requested model, trains it, and writes:
  - a checkpoint into `checkpoints/`
  - a resolved copy of the config into `configs/`

- [test.py](test.py)
  Evaluation entrypoint. Rebuilds the model from the saved YAML, runs the test set, and optionally performs automated feature ablations.

- [Simulator_start_dataset.py](Simulator_start_dataset.py)
  Initializes the simulator from states taken directly from the dataset and compares simulated closed-loop rollouts against reference commands/energy.

- [Simulator_random_start.py](Simulator_random_start.py)
  Samples random physically plausible initial states and measures convergence, energy, and time-to-target.

- [Simulator_race_drone.py](Simulator_race_drone.py)
  Re-centers the drone around successive gates and evaluates repeated gate-passing behavior.

- [Simulator_start_dataset_C.py](Simulator_start_dataset_C.py), [Simulator_random_start_C.py](Simulator_random_start_C.py), [Simulator_race_drone_C.py](Simulator_race_drone_C.py)
  C-backed versions of the simulators. They use exported controllers from `C_codes/<model>/` through `ctypes`.

- [C_codes](C_codes)
  C exports for the trained checkpoints. Each model folder contains `nn_parameters.*`, `nn_operations.*`, `run_controller.c`, `test_controller.c`, and `compare_python_c.py`.

- [utils/model_builder.py](utils/model_builder.py)
  Centralized architecture factory used by all scripts. This is the main place where model type, width scaling, preprocessing blocks, and CfC/LTC options are interpreted.

- [utils/quadrotor_sim.py](utils/quadrotor_sim.py)
  Shared simulation code:
  - state transforms
  - continuous-time dynamics
  - numerical integration
  - observation window management
  - checkpoint-backed controller rollout

- [utils/quadrotor_sim_c.py](utils/quadrotor_sim_c.py)
  Shared rollout code for the C-backed simulators.

- [utils/c_controller.py](utils/c_controller.py)
  Compiles and loads the exported C controllers as shared libraries and exposes `nn_reset`/`nn_control` from Python.

- [utils/ablation.py](utils/ablation.py)
  Feature-group masking for automated testing ablations.

- [utils/feedforward.py](utils/feedforward.py)
  Plain MLP baseline, wrapped to match the same sequence interface as the recurrent models.

- [utils/data.py](utils/data.py)
  Dataset loading, normalization-vector construction, feature expansion, and sliding-window generation.

- [utils/lightning.py](utils/lightning.py)
  Shared Lightning wrapper used by training and test-time checkpoint execution.

## Main Workflow

### 1. Training

Edit [train_config.yaml](train_config.yaml), then run from the repository root:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/train.py
```

Outputs:

- `LNN_behavioural_cloning_quadrotor/checkpoints/<checkpoint>.ckpt`
- `LNN_behavioural_cloning_quadrotor/configs/<checkpoint>.yaml`

The saved YAML is important because test/simulators rebuild the exact same architecture from it.

### 2. Testing

Set `model_path` in [test_config.yaml](test_config.yaml) to the checkpoint stem, then run:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/test.py
```

Optional plot:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/test.py --plot
```

### 3. Simulators

Set `model_path` in [simulator_config.yaml](simulator_config.yaml), then choose one.

Python/PyTorch controllers:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/Simulator_start_dataset.py
.venv/bin/python LNN_behavioural_cloning_quadrotor/Simulator_random_start.py
.venv/bin/python LNN_behavioural_cloning_quadrotor/Simulator_race_drone.py
```

C-exported controllers:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/Simulator_start_dataset_C.py
.venv/bin/python LNN_behavioural_cloning_quadrotor/Simulator_random_start_C.py
.venv/bin/python LNN_behavioural_cloning_quadrotor/Simulator_race_drone_C.py
```

The C-backed simulators automatically map `simulator_config.yaml -> model_path` to the matching folder in `C_codes`. To force a specific C export:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/Simulator_random_start_C.py \
  --c-model-dir LNN_behavioural_cloning_quadrotor/C_codes/GRU
```

### Visualization

A helper script is available to animate one or more dataset-based rollouts in the same window.

PyTorch controller visualization:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/visualize_rollout.py \
  --trajectory 0 --trajectories 4 --simultaneous --draw-path
```

C-exported controller visualization:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/visualize_rollout_C.py \
  --trajectory 0 --trajectories 4 --simultaneous --draw-path
```

This will:

- simulate trajectories `0..3`
- draw them together in one animation
- show the path of each drone

If you want to save the animation instead of opening a window:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/visualize_rollout.py \
  --trajectory 0 --trajectories 4 --record --output /tmp/rollout.mp4
```

For the C version:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/visualize_rollout_C.py \
  --trajectory 0 --trajectories 4 --record --output /tmp/rollout_c.mp4
```

The C visualizer uses `simulator_config.yaml -> model_path` to pick a folder in `C_codes`. To force a specific export:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/visualize_rollout_C.py \
  --c-model-dir LNN_behavioural_cloning_quadrotor/C_codes/CFC \
  --trajectory 0 --trajectories 4 --simultaneous --draw-path
```

### Benchmarks

There are two benchmark styles.

`benchmark_python_vs_c.py` keeps the simulator loop in Python and calls the C controller through `ctypes`. This is useful for checking integration overhead, but it is not representative of firmware C:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/benchmark_python_vs_c.py --runs 1000
```

For a faster exploratory run:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/benchmark_python_vs_c.py \
  --runs 100 --horizon-steps 100
```

The benchmark measures the complete Python simulation loop. The C path still crosses the Python/C `ctypes` boundary once per timestep, so this is not the same as running the whole controller and dynamics loop in firmware C.

For a cleaner comparison, run the PyTorch/Python benchmark and the full-C benchmark separately. The full-C benchmark currently exists for the MLP export:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/benchmark_python_only.py \
  --runs 1000 --horizon-steps 400

gcc -std=c99 -O3 -Wall -Wextra \
  LNN_behavioural_cloning_quadrotor/C_codes/MLP/benchmark_full_c.c \
  LNN_behavioural_cloning_quadrotor/C_codes/MLP/nn_operations.c \
  LNN_behavioural_cloning_quadrotor/C_codes/MLP/nn_parameters.c \
  -lm -o /tmp/benchmark_full_c_mlp

/tmp/benchmark_full_c_mlp 1000 400 0.01
```

On the MLP benchmark with 1000 rollouts and 400 steps each, the measured times were:

```text
PyTorch/Python: 385.625678 s total, 0.000964064 s/step
Full C:           5.176626 s total, 0.000012942 s/step
```

### C Controller Exports

All exported C controllers live in [C_codes](C_codes). The available folders are:

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

Each folder contains:

- `nn_parameters.h/.c`: exported weights, biases, and normalization limits
- `nn_operations.h/.c`: C inference implementation
- `run_controller.c`: command-line runner that receives the raw input vector
- `test_controller.c`: fixed-input C smoke test
- `compare_python_c.py`: numerical comparison against the original PyTorch checkpoint

To compare one exported model against PyTorch:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/C_codes/CFC/compare_python_c.py
```

To compile and run the standalone C smoke test:

```bash
cd LNN_behavioural_cloning_quadrotor/C_codes/CFC
gcc -std=c99 -Wall -Wextra test_controller.c nn_operations.c nn_parameters.c -lm -o test_controller
./test_controller
```

For recurrent and liquid models, `nn_control` keeps hidden state in static storage. Call `nn_reset()` before starting a new trajectory or whenever the controller memory should be cleared.

## Supported Model Options

The primary architecture switch lives in `train_config.yaml -> model.type`.

Supported values:

- `cfc`
- `ltc`
- `ncp`
- `ctrnn`
- `simplernn`
- `gru`
- `lstm`
- `mlp`

### CfC-specific options

When `model.type: cfc`, the following are used:

- `model.cfc_mode`
  Values: `default`, `pure`, `no_gate`

- `model.backbone_units`
- `model.backbone_layers`
- `model.backbone_dropout`

### NCP-specific options

When `model.type: ncp`, the following are used:

- `model.ncp.inter_neurons`
- `model.ncp.command_neurons`
- `model.ncp.sensory_fanout`
- `model.ncp.inter_fanout`
- `model.ncp.recurrent_command_synapses`
- `model.ncp.motor_fanin`

The refactored pipeline builds NCP controllers as `CfC` models with an
`ncps.wirings.NCP(...)` wiring.

### Global network scaling

Use:

- `model.scale_factor`

This scales:

- recurrent hidden width
- CfC backbone width
- MLP preprocessing widths
- feedforward hidden widths
- convolution output width

### Recurrent neuron count

Use:

- `model.no_neurons_layer`

This controls the hidden width for recurrent backbones.

### Feedforward / NN baseline

To use a plain non-recurrent controller:

```yaml
model:
  type: mlp
  hidden_layers: [128, 128]
  activation: relu
```

This follows the same train/test/simulator path as the recurrent models.

## Preprocessing Blocks

### Convolutional preprocessing

Use:

```yaml
conv_block:
  value: true
  output_dim: 256
```

This requires sequencing:

```yaml
sequencing:
  value: true
  seq_len: 1
```

### MLP preprocessing

Use:

```yaml
mlp_block:
  value: true
  no_layers: [64, 128, 256]
```

This inserts an MLP feature extractor before the recurrent core.

## Automated Ablation Testing

Ablation is configured in [test_config.yaml](test_config.yaml):

```yaml
ablation:
  enabled: true
  fill_value: 0.0
  feature_sets:
    position: [dx, dy, dz]
    velocity: [vx, vy, vz]
    attitude: [phi, theta, psi]
```

For each named group:

- the matching feature indices are resolved from the configured input label order
- the input tensor is copied
- those channels are replaced with `fill_value`
- the test evaluation is rerun

## Configuration Files

### `train_config.yaml`

Controls:

- dataset labels and paths
- dataloader settings
- sequencing
- optional preprocessing blocks
- model family and hyperparameters
- scaling
- logging

### `test_config.yaml`

Controls:

- which saved model to load
- which test dataset to evaluate
- whether to run ablations
- plotting toggle

### `simulator_config.yaml`

Controls:

- which saved model to load
- which simulation horizon and timestep to use
- which integration method to use
- convergence thresholds
- random-start simulation count

## Notes On Legacy Files

This refactor focuses on the main supervised-learning path. Older side scripts such as:

- `*_NN.py`
- `Simulator_start_dataset_model_comparison.py`
- `test_NN.py`

were left as legacy utilities and were not migrated onto the new shared helper stack.

## Typical Usage Pattern

1. Train with `train.py`.
2. Copy the resulting checkpoint stem.
3. Put that stem into `test_config.yaml` and/or `simulator_config.yaml`.
4. Run `test.py` for baseline and ablation metrics.
5. Run one or more simulator scripts for closed-loop behavior checks.
