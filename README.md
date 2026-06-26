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

Most Bebop2 checks should start with the C-exported square simulator. It runs the supervised-learning controller from `C_codes/` while changing only the simulated plant through `--dynamics-model`.

For the current CfC-on-Bebop2 action plot:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_gazebo_square_C \
  --model CFC \
  --dynamics-model quadrotor_sim_matlab \
  --time-simulation 20 \
  --dist-error 0.001 \
  --plot-actions \
  --no-animation
```

Outputs are saved in [simulators/runs](simulators/runs):

- `cfc_sl_bebop2_actions.png`
- `cfc_sl_bebop2_actions.csv`

The plot format matches the RL action plots: four stacked motor traces in RPM, plus CSV columns for motor RPM and RPM deltas. You do not need to set `MPLCONFIGDIR`; the simulator sets Matplotlib's cache/config directory to `simulators/runs/.matplotlib` before importing Matplotlib.

For an animated rollout with the same simulator:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_gazebo_square_C
```

Useful options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `CFC` | C export to run: `MLP`, `LTC`, `RNN`, `CFC`, `CFC_PURE`, `GRU`, `LSTM`, `NCP_CFC`, etc. |
| `--dynamics-model` | `quadrotor_sim_matlab` | Bebop2 MATLAB force/moment model. Use `quadrotor_sim_original` only for the old reduced model. |
| `--time-simulation` | `60.0` | Maximum simulated time in seconds. |
| `--dist-error` | `0.1` | Waypoint switch radius in meters. |
| `--plot-actions` | disabled | Save RL-style motor plots and CSV under `simulators/runs`. |
| `--no-animation` | disabled | Run metrics/plots without opening the OpenCV animation window. |
| `--record --output <file.mp4>` | disabled | Save the animation video. |

The square route is hard-coded as four ENU waypoints:
`(2.0, 1.5, 1.5)`, `(2.0, -1.5, 1.5)`, `(-2.0, -1.5, 1.5)`, `(-2.0, 1.5, 1.5)`.

Other simulator entrypoints still exist for older workflows:

| Use case | Python/PyTorch | C export |
| --- | --- | --- |
| Dataset start state | `Simulator_start_dataset` | `Simulator_start_dataset_C` |
| Random starts | `Simulator_random_start` | `Simulator_random_start_C` |
| Race/gate rollout | `Simulator_race_drone` | `Simulator_race_drone_C` |
| Dataset rollout visualizer | `visualize_rollout` | `visualize_rollout_C` |
| Figure-eight C route | n/a | `Simulator_gazebo_figure8_C` |

Example:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_random_start_C
```

These older scripts use [simulator_config.yaml](simulator_config.yaml) unless they expose a CLI override. The C visualizer and C simulators can usually be pointed at a different export with `--c-model-dir LNN_behavioural_cloning_quadrotor/C_codes/<MODEL>`.

### Benchmarks

Benchmark scripts are available, but they are separate from normal simulation:

- `benchmark_python_vs_c_ctypes.py` compares PyTorch vs C-controller calls while keeping the simulator loop in Python.
- `benchmark_python_only.py` measures the Python/PyTorch simulator path.
- `C_codes/MLP/benchmark_full_c.c` is the standalone full-C benchmark currently available for MLP.

Example:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.benchmark_python_vs_c_ctypes \
  --runs 100 --horizon-steps 100
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
