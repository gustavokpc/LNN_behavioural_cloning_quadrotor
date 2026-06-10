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

## Project Layout

- [train.py](train.py): trains a controller from `train_config.yaml` and saves the checkpoint/config pair.
- [test.py](test.py): rebuilds a saved checkpoint, evaluates it, and optionally runs ablations.
- [simulators](simulators): closed-loop rollout scripts, including the Gazebo/Paparazzi C-controller comparisons.
- [rl](rl): PPO/SB3 workspace for Bebop2 experiments on top of the trained CfC controller.
- [C_codes](C_codes): exported C controllers for each trained checkpoint.
- [checkpoints](checkpoints) and [configs](configs): saved weights and matching YAML configs.
- [utils](utils): shared model-building, data, dynamics, controller-loading, simulation, and plotting utilities.

## Main Workflow

The command examples below assume you are in the parent folder `LNN_estag`:

```bash
cd /home/gustavokpc/Documents/ESTAG/LNN_estag
```

### 1. Training

Edit [train_config.yaml](train_config.yaml), then run:

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

### 3. Simulations

Recommended Bebop2/Gazebo square simulation with the trained CfC C export:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_gazebo_square_C \
  --model CFC \
  --dynamics-model quadrotor_sim_matlab \
  --time-simulation 60 \
  --dist-error 0.1 \
  --dt 0.01 \
  --integration-method rk4 \
  --implicit-iters 1 \
  --start-waypoint-index 3 \
  --start-alt 1.0 \
  --waypoint-alt 1.5 \
  --auto-play
```

The same defaults are already built into the command-line parser, so this shorter command is equivalent for the square simulation:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_gazebo_square_C
```

Gazebo/Paparazzi square reads `NN_SQ_*` waypoints from `nn_waypoints_square.xml`. The figure-eight version reads `RL_F8_1..8` from `rl_cfc_waypoints_square.xml`:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_gazebo_figure8_C
```

Useful options for `Simulator_gazebo_square_C` and `Simulator_gazebo_figure8_C`:

| Option | Default if omitted | Choices / meaning |
| --- | --- | --- |
| `--model` | `CFC` | `MLP`, `LTC`, `RNN`, `CONV_CFC_DEFAULT`, `CFC`, `CFC_PURE`, `CTRNN`, `GRU`, `LSTM`, `NCP_CFC` |
| `--dynamics-model` | `quadrotor_sim_matlab` | `quadrotor_sim` for the original reduced model, `quadrotor_sim_matlab` for the Bebop2 MATLAB force/moment model |
| `--time-simulation` | `60.0` | Maximum simulated time in seconds |
| `--dist-error` | `0.1` | Waypoint switching distance in meters |
| `--dt` | `0.01` | Simulation timestep in seconds |
| `--integration-method` | `rk4` | Integration method passed to the rollout code |
| `--implicit-iters` | `1` | Iterations used by implicit integration methods |
| `--start-waypoint-index` | `3` for square, `0` for figure-eight | First waypoint index in the route |
| `--start-alt` | `1.0` | Initial `STDBY` altitude in meters |
| `--waypoint-alt` | `1.5` | Target waypoint altitude in meters |
| `--auto-play` / `--no-auto-play` | `--auto-play` | Start the animation automatically or wait for manual play |
| `--reset-each-waypoint` | disabled | Reset recurrent/CfC controller memory at each waypoint |
| `--no-animation` | disabled | Run metrics without opening the animation |
| `--record --output <file.mp4>` | disabled, `gazebo_square_cfc.mp4` or `gazebo_figure8_cfc.mp4` | Save the animation instead of only displaying it |
| `--flight-plan <path>` | script-specific Paparazzi XML path | Use a different waypoint XML |
| `--model-config <path>` | matching YAML from `MODEL_PRESETS` | Override the config YAML for the selected model |
| `--c-model-dir <path>` | matching folder from `MODEL_PRESETS` | Override the exported C controller folder |

Dynamics equations live in [utils/dynamics_models](utils/dynamics_models). The neural-network normalization limits stay fixed to the training data; only the simulated plant equation changes when you switch `--dynamics-model`.

Other closed-loop simulators use [simulator_config.yaml](simulator_config.yaml) for `model_path`, horizon, thresholds, and dataset paths.

Python/PyTorch controllers:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_start_dataset
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_random_start
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_race_drone
```

C-exported controllers:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_start_dataset_C
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_random_start_C
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_race_drone_C
```

To force a specific C export in those C-backed simulators:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_random_start_C \
  --c-model-dir LNN_behavioural_cloning_quadrotor/C_codes/GRU
```

### Visualization

A helper script is available to animate one or more dataset-based rollouts in the same window.

PyTorch controller visualization:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.visualize_rollout \
  --trajectory 0 --trajectories 4 --simultaneous --draw-path
```

C-exported controller visualization:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.visualize_rollout_C \
  --trajectory 0 --trajectories 4 --simultaneous --draw-path
```

This will:

- simulate trajectories `0..3`
- draw them together in one animation
- show the path of each drone

If you want to save the animation instead of opening a window:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.visualize_rollout \
  --trajectory 0 --trajectories 4 --record --output /tmp/rollout.mp4
```

For the C version:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.visualize_rollout_C \
  --trajectory 0 --trajectories 4 --record --output /tmp/rollout_c.mp4
```

The C visualizer uses `simulator_config.yaml -> model_path` to pick a folder in `C_codes`. To force a specific export:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.visualize_rollout_C \
  --c-model-dir LNN_behavioural_cloning_quadrotor/C_codes/CFC \
  --trajectory 0 --trajectories 4 --simultaneous --draw-path
```

### Benchmarks

There are two benchmark styles.

`simulators/benchmark_python_vs_c_ctypes.py` keeps the simulator loop in Python and calls the C controller through `ctypes`. This is useful for checking integration overhead, but it is not representative of firmware C:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.benchmark_python_vs_c_ctypes --runs 1000
```

For a faster exploratory run:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.benchmark_python_vs_c_ctypes \
  --runs 100 --horizon-steps 100
```

The benchmark measures the complete Python simulation loop. The C path still crosses the Python/C `ctypes` boundary once per timestep, so this is not the same as running the whole controller and dynamics loop in firmware C.

For a cleaner comparison, run the PyTorch/Python benchmark and the full-C benchmark separately. The full-C benchmark currently exists for the MLP export:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.benchmark_python_only \
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
