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

Most checks should start with the C-exported square simulator. It runs a supervised-learning controller from `C_codes/` and lets you choose the simulated plant, motor time constant, waypoint tolerance, input noise, and plots from the command line.

The general command shape is:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_gazebo_square_C \
  --model <MODEL_PRESET> \
  --dynamics-model quadrotor_sim_matlab \
  --time-simulation 20 \
  --dist-error 0.001 \
  --tau 0.03 \
  --plot-signals
```

Preset models available through `--model`:

| `--model` | Config | C export |
| --- | --- | --- |
| `CFC` | `configs/new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.yaml` | `C_codes/CFC` |
| `CFC_PURE` | `configs/new_CFC_pure_64_neurons_seq_1_epoch=17_val_loss=0.000203.yaml` | `C_codes/CFC_PURE` |
| `CONV_CFC_DEFAULT` | `configs/conv_cfc_default_n64_epoch=17_val_loss=0.000326.yaml` | `C_codes/CONV_CFC_DEFAULT` |
| `NCP_CFC` | `configs/new_NCP_CFC_60_neurons_seq_1_epoch=18_val_loss=0.000143.yaml` | `C_codes/NCP_CFC` |
| `LTC` | `configs/LTC_64_neurons_seq_1_epoch=18_val_loss=0.000193.yaml` | `C_codes/LTC` |
| `CTRNN` | `configs/new_CTRNN_64_neurons_seq_1_epoch=19_val_loss=0.000150.yaml` | `C_codes/CTRNN` |
| `RNN` | `configs/RNN_64_neurons_seq_1_epoch=17_val_loss=0.000147.yaml` | `C_codes/RNN` |
| `GRU` | `configs/new_GRU_64_neurons_seq_1_epoch=19_val_loss=0.000088.yaml` | `C_codes/GRU` |
| `LSTM` | `configs/new_LSTM_64_neurons_seq_1_epoch=17_val_loss=0.000092.yaml` | `C_codes/LSTM` |
| `MLP` | `configs/mlp_epoch=19_val_loss=0.003130.yaml` | `C_codes/MLP` |

For example, `--model CFC` launches the Bebop1 CfC export in `C_codes/CFC`; `--model GRU` launches the Bebop1 GRU export; and so on. These presets are convenient when the config and C folder are already listed above.

The Bebop2 convolutional CfC export is not a `--model` preset, so pass the config and C folder explicitly:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_gazebo_square_C \
  --model-config LNN_behavioural_cloning_quadrotor/configs/bbp2_conv_cfc_default_n64_epoch=19_val_loss=0.000143.yaml \
  --c-model-dir LNN_behavioural_cloning_quadrotor/C_codes/BBP2_CONV_CFC \
  --dynamics-model quadrotor_sim_matlab \
  --time-simulation 20 \
  --dist-error 0.001 \
  --tau 0.03 \
  --input-noise-p-sigma 0.24 \
  --input-noise-q-sigma 0.12 \
  --input-noise-r-sigma 0.10 \
  --input-noise-seed 123 \
  --reset-each-waypoint \
  --plot-signals \
  --signals-plot-output LNN_behavioural_cloning_quadrotor/simulators/runs/bbp2_conv_cfc_new_tau003_dist0001_pqr_noise_all_signals.png
```

This launches a 20 s square simulation with the Bebop2 CONV-CfC C model, the MATLAB-style quadrotor dynamics, `tau = 0.03`, waypoint tolerance `dist_error = 0.001`, Gaussian input noise on `p/q/r`, controller reset at each waypoint, and the all-signals plot enabled.

To run the Bebop1 CfC model with the same simulation settings, use the preset instead of explicit Bebop2 paths:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_gazebo_square_C \
  --model CFC \
  --dynamics-model quadrotor_sim_matlab \
  --time-simulation 20 \
  --dist-error 0.001 \
  --tau 0.03 \
  --input-noise-p-sigma 0.24 \
  --input-noise-q-sigma 0.12 \
  --input-noise-r-sigma 0.10 \
  --input-noise-seed 123 \
  --reset-each-waypoint \
  --plot-signals \
  --signals-plot-output LNN_behavioural_cloning_quadrotor/simulators/runs/cfc_bebop1_tau003_dist0001_pqr_noise_all_signals.png
```

Useful square-simulator options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `CFC` | Selects one of the preset config/C-export pairs. |
| `--model-config <yaml>` | preset value | Overrides the YAML config. Use this for exports such as `BBP2_CONV_CFC`. |
| `--c-model-dir <dir>` | preset value | Overrides the C controller folder. Use this together with `--model-config` for custom exports. |
| `--dynamics-model` | `quadrotor_sim_matlab` | Plant model. Use `quadrotor_sim_matlab` for the MATLAB-style quadrotor model, or `quadrotor_sim_original` for the older reduced model. |
| `--dt` | `0.01` | Simulation time step in seconds. |
| `--time-simulation` | `60.0` | Maximum simulated time in seconds. |
| `--dist-error` | `0.1` | Waypoint switch radius in meters. |
| `--tau` | dynamics default | Runtime motor time-constant override in seconds, for example `--tau 0.03`. |
| `--input-noise-p-sigma` | `0.0` | Gaussian input noise sigma for body rate `p` in rad/s. |
| `--input-noise-q-sigma` | `0.0` | Gaussian input noise sigma for body rate `q` in rad/s. |
| `--input-noise-r-sigma` | `0.0` | Gaussian input noise sigma for body rate `r` in rad/s. |
| `--input-noise-seed` | none | Fixed random seed for repeatable input noise. |
| `--reset-each-waypoint` | disabled | Calls `nn_reset()` whenever the waypoint changes, useful for recurrent/liquid controllers. |
| `--plot-actions` | disabled | Save RL-style motor plots and CSV under `simulators/runs`. |
| `--action-plot-output <png>` | `simulators/runs/cfc_sl_bebop2_actions.png` | Output path for `--plot-actions`. |
| `--plot-signals` | disabled | Save the all-signals plot with states, commands, errors, and motor traces. |
| `--signals-plot-output <png>` | `simulators/runs/square_all_state_commands.png` | Output path for `--plot-signals`. |
| `--no-animation` | disabled | Run metrics/plots without opening the OpenCV animation window. |
| `--record --output <file.mp4>` | disabled | Save the animation video. |
| `--start-waypoint-index` | `3` | Which square waypoint index to start from. |
| `--start-alt` | `1.0` | Initial altitude in meters. |
| `--waypoint-alt` | `1.5` | Square waypoint altitude in meters. |

The square route is hard-coded as four ENU waypoints:
`(2.0, 1.5, 1.5)`, `(2.0, -1.5, 1.5)`, `(-2.0, -1.5, 1.5)`, `(-2.0, 1.5, 1.5)`.

Outputs are saved in [simulators/runs](simulators/runs) unless another output path is supplied. You do not need to set `MPLCONFIGDIR`; the simulator sets Matplotlib's cache/config directory to `simulators/runs/.matplotlib` before importing Matplotlib.

For an animated rollout with default settings:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.simulators.Simulator_gazebo_square_C
```

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
BBP2_CONV_CFC
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
RECURRENT_PPO_57600000
FIGURE8_GATES_RECURRENT_PPO
FIGURE8_GATES_LTC_RECURRENT_PPO
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
