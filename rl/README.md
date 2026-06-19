# RL Workspace

This folder is reserved for PPO/SB3 experiments that fine-tune or wrap the trained CfC controller on the Bebop2 MATLAB dynamics.

## Imported Legacy PPO Code

The previous PPO/SB3 training code has been imported under [legacy_ppo](legacy_ppo). The goal of this first step is to keep it as close as possible to the original version while making it importable from this repository.

Imported files:

- [legacy_ppo/drone_ppo_sb3.py](legacy_ppo/drone_ppo_sb3.py): original PPO/RecurrentPPO training script, custom CfC policy, callbacks, rendering, and CLI.
- [legacy_ppo/quadcopter_envs.py](legacy_ppo/quadcopter_envs.py): original vectorized gates environment and 8-shaped gate trajectory.
- [legacy_ppo/quadcopter_hover_envs.py](legacy_ppo/quadcopter_hover_envs.py): original hover environment dependency.
- [legacy_ppo/quadcopter_animation](legacy_ppo/quadcopter_animation): original OpenCV renderer used by `--render`.

Minimal changes made during import:

- Imports were changed to relative package imports, for example `.quadcopter_envs`.
- Checkpoints now save under `rl/checkpoints/legacy_ppo/`.
- TensorBoard logs now default to `rl/runs/legacy_ppo/`.

The legacy environment still uses its original observation vector and its original action space:

```text
action_space = Box(low=-1, high=u_lim, shape=(4,))
```

That `[-1, 1]` range is not forced by PPO itself. It comes from the Gym/SB3 environment definition. PPO samples actions according to the environment's `action_space`; here the old environment deliberately normalized motor commands to `[-1, 1]`, then converted them internally to motor commands/speeds.

To run the legacy trainer from the parent folder `LNN_estag` after installing the RL dependencies:

```bash
.venv/bin/python -m pip install -r LNN_behavioural_cloning_quadrotor/rl/legacy_ppo/requirements.txt
```

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.legacy_ppo.drone_ppo_sb3 \
  --policy-type recurrent_ppo \
  --num-envs 100 \
  --total-timesteps 500000
```

For a very small smoke run:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.legacy_ppo.drone_ppo_sb3 \
  --policy-type ppo \
  --num-envs 2 \
  --rollout-fragment-length 32 \
  --batch-size 64 \
  --total-timesteps 128 \
  --checkpoint-freq 128
```

This imported legacy code is not yet the final Bebop2 square-training environment. The next adaptation step is to replace the old 8-shaped gate definition and old symbolic dynamics with this repository's `quadrotor_sim_matlab` Bebop2 dynamics and square waypoints.

## Bebop2 Waypoint Environment

The first new environment is [envs/bebop2_waypoints_env.py](envs/bebop2_waypoints_env.py). It is separate from `legacy_ppo/` so the old implementation remains available as a reference.

Current contract:

- Observation space: 19 values, matching the supervised-learning state interface:
  `dx, dy, dz, vx, vy, vz, phi, theta, psi, p, q, r, Mx_ext, My_ext, Mz_ext, omega1, omega2, omega3, omega4`.
- Action space: 4 motor commands normalized in `[0, 1]`, matching the supervised-learning output convention.
- Dynamics: `utils.dynamics_models.quadrotor_sim_matlab`.
- Targets: points in space, not physical gates.
- Default square points: `(2.0, 1.5, -1.5)`, `(2.0, -1.5, -1.5)`, `(-2.0, -1.5, -1.5)`, `(-2.0, 1.5, -1.5)`.
- Waypoint switching: distance threshold, not gate-plane crossing.

This environment is the intended base for the new PPO/SB3 training path.

The matching PPO/SB3 entrypoint is [drone_ppo_bebop2.py](drone_ppo_bebop2.py). It uses `Bebop2WaypointEnv`, not the legacy `Quadcopter3DGates` environment.

All defaults for the new RL trainer live directly in `parse_args()` inside [drone_ppo_bebop2.py](drone_ppo_bebop2.py). There is no YAML config for this path right now.

Full hot-start training command with the current defaults written explicitly:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --policy-type bc_ppo \
  --bc-config LNN_behavioural_cloning_quadrotor/configs/new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.yaml \
  --bc-checkpoint new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.ckpt \
  --num-envs 32 \
  --seed 0 \
  --learning-rate 0.0003 \
  --rollout-fragment-length 512 \
  --batch-size 1024 \
  --gamma 0.999 \
  --lam 0.95 \
  --clip-param 0.2 \
  --entropy-coeff 0.01 \
  --vf-coeff 0.5 \
  --n-epochs 10 \
  --total-timesteps 1000000 \
  --checkpoint-freq 100000 \
  --dt 0.01 \
  --max-steps 6000 \
  --waypoint-radius 0.2 \
  --integration-method rk4 \
  --implicit-iters 1 \
  --device auto
```

Because those values are defaults, the short equivalent is:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --policy-type bc_ppo
```

Small smoke run:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --policy-type ppo \
  --num-envs 2 \
  --rollout-fragment-length 32 \
  --batch-size 64 \
  --total-timesteps 128 \
  --checkpoint-freq 128
```

Longer recurrent CfC-policy run:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --policy-type recurrent_ppo \
  --num-envs 32 \
  --total-timesteps 1000000
```

Hot-start PPO from the supervised CfC checkpoint:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --policy-type bc_ppo \
  --num-envs 32 \
  --total-timesteps 1000000
```

This loads the supervised-learning checkpoint as the PPO actor initialization, trains a copy of those parameters, and saves a new SB3 `.zip` under `rl/checkpoints/bebop2_waypoints/bc_ppo/`. It does not modify the original SL `.ckpt`.

Residual PPO around the supervised CfC checkpoint:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --policy-type residual_ppo \
  --residual-scale 0.05 \
  --num-envs 32 \
  --total-timesteps 1000000
```

In `residual_ppo`, the supervised CfC is loaded as a frozen base controller. PPO does not overwrite the CfC parameters. Instead, PPO outputs a residual action in `[-1, 1]`, and the environment applies:

```text
final_motor_command = clip(cfc_motor_command + residual_scale * ppo_residual, 0, 1)
```

The resulting SB3 policy is saved under `rl/checkpoints/bebop2_waypoints/residual_ppo/`. To make the correction more conservative, reduce `--residual-scale`, for example `0.02`. To let PPO correct more aggressively, increase it, for example `0.10`.

The default supervised checkpoint/config used by `bc_ppo` are:

```text
--bc-config LNN_behavioural_cloning_quadrotor/configs/new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.yaml
--bc-checkpoint new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.ckpt
```

Visualize a trained checkpoint:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --render \
  --cont LNN_behavioural_cloning_quadrotor/rl/checkpoints/bebop2_waypoints/ppo/ppo_bebop2_waypoints.zip \
  --policy-type ppo
```

Visualize several attempts one after another in the same viewer. Use `J`/`L` to switch attempts:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --render \
  --render-episodes 5 \
  --cont LNN_behavioural_cloning_quadrotor/rl/checkpoints/bebop2_waypoints/ppo/ppo_bebop2_waypoints.zip \
  --policy-type ppo
```

Visualize several attempts at the same time:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --render \
  --render-episodes 5 \
  --simultaneous \
  --cont LNN_behavioural_cloning_quadrotor/rl/checkpoints/bebop2_waypoints/ppo/ppo_bebop2_waypoints.zip \
  --policy-type ppo
```

Record the rollout instead of only opening the interactive viewer:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --render \
  --record \
  --cont LNN_behavioural_cloning_quadrotor/rl/checkpoints/bebop2_waypoints/ppo/ppo_bebop2_waypoints.zip \
  --policy-type ppo \
  --output /tmp/bebop2_ppo_rollout.mp4
```

## Bebop2 Figure-8 Gates Environment

The same trainer also has a legacy-style figure-8 gates mode:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --track figure8_gates \
  --policy-type recurrent_ppo \
  --num-envs 32 \
  --total-timesteps 1000000
```

This mode uses [envs/bebop2_figure8_gates_env.py](envs/bebop2_figure8_gates_env.py), which keeps the legacy gate task contract while replacing the old symbolic dynamics with `quadrotor_sim_matlab`:

- figure-8 `gate_pos` and `gate_yaw` from the legacy PPO environment;
- observation layout matching the legacy gate input: 16-state relative-to-gate core plus future gate features/history options;
- action space in the legacy `[-1, 1]` convention;
- internal conversion to Bebop2 motor commands in `[0, 1]`;
- Bebop2 19-state integration through `integrate_state(...)` after selecting `quadrotor_sim_matlab`;
- gate-plane pass/collision logic, ground collision, out-of-bounds checks, and rollout metrics.

Because `figure8_gates` uses the legacy-style observation vector instead of the 19-value supervised-learning input, use `--policy-type ppo` or `--policy-type recurrent_ppo`. The `bc_ppo` and `residual_ppo` modes remain available for the waypoint environment, but are intentionally rejected for `figure8_gates`.

Small smoke run:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --track figure8_gates \
  --policy-type ppo \
  --num-envs 2 \
  --rollout-fragment-length 8 \
  --batch-size 16 \
  --total-timesteps 16 \
  --checkpoint-freq 16 \
  --max-steps 20 \
  --device cpu
```

Checkpoints are saved under `rl/checkpoints/figure8_gates/<policy-type>/`, and TensorBoard logs default to `rl/runs/figure8_gates/`.

Render a trained figure-8 checkpoint with the legacy gate viewer:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --track figure8_gates \
  --render \
  --cont LNN_behavioural_cloning_quadrotor/rl/checkpoints/figure8_gates/recurrent_ppo/recurrent_ppo_figure8_gates.zip \
  --policy-type recurrent_ppo
```

### Trainer Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--policy-type` | `recurrent_ppo` | `ppo` trains a feedforward MLP policy from scratch; `recurrent_ppo` trains the legacy SB3 CfC recurrent policy from scratch; `bc_ppo` initializes the PPO actor from the supervised CfC checkpoint; `residual_ppo` freezes the supervised CfC and trains a PPO correction around it. |
| `--num-envs` | `32` | Number of parallel vectorized environments. |
| `--seed` | `0` | Random seed. |
| `--cell-size` | `64` | Hidden size for `recurrent_ppo`. |
| `--learning-rate` | `0.0003` | PPO optimizer learning rate. |
| `--max-log-std` | `1.0` | Maximum Gaussian policy log standard deviation. |
| `--rollout-fragment-length` | `512` | PPO `n_steps`: rollout steps per environment before each update. |
| `--batch-size` | `1024` | Minibatch size for PPO optimization. |
| `--gamma` | `0.999` | Discount factor. |
| `--lam` | `0.95` | GAE lambda. |
| `--clip-param` | `0.2` | PPO clipping range. |
| `--entropy-coeff` | `0.01` | Entropy bonus coefficient. |
| `--vf-coeff` | `0.5` | Value-function loss coefficient. |
| `--total-timesteps` | `1000000` | Total training timesteps. |
| `--checkpoint-freq` | `100000` | Frequency for intermediate SB3 checkpoint saves. |
| `--tensorboard-log` | `rl/runs/bebop2_waypoints` | TensorBoard log directory. |
| `--device` | `auto` | SB3/PyTorch device: `auto`, `cpu`, or CUDA device string. |
| `--n-epochs` | `10` | PPO optimization epochs per rollout. |
| `--cfc-timespan` | `0.01` | Timespan used by the legacy recurrent CfC policy. |
| `--use-flatten-features` | `True` | Feature extractor choice for `recurrent_ppo`. |
| `--bc-config` | `configs/new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.yaml` | YAML used to rebuild the supervised CfC for `bc_ppo`. |
| `--bc-checkpoint` | `new_CFC_64_neurons_seq_1_epoch=18_val_loss=0.000142.ckpt` | Supervised CfC checkpoint used to initialize `bc_ppo`. |
| `--bc-value-hidden-dim` | `64` | Critic hidden size used by `bc_ppo`. |
| `--residual-scale` | `0.05` | Residual correction magnitude for `residual_ppo`; the final action is `clip(CfC + residual_scale * PPO, 0, 1)`. |
| `--cont` | empty | Continue from an existing SB3 `.zip` checkpoint. |
| `--dt` | `0.01` | Environment timestep. |
| `--max-steps` | `6000` | Maximum steps per episode. |
| `--track` | `square_waypoints` | `square_waypoints` keeps the current Bebop2 waypoint trainer; `figure8_gates` uses legacy-style figure-8 gates with Bebop2 dynamics. |
| `--waypoint-radius` | `0.2` | Distance threshold to mark a waypoint reached. |
| `--gate-size` | `1.5` | Gate pass/collision box size for `figure8_gates`. |
| `--gates-ahead` | `1` | Number of future gates appended to the legacy-style observation in `figure8_gates`. |
| `--integration-method` | `rk4` | Dynamics integration method. |
| `--implicit-iters` | `1` | Iterations for implicit integration. |
| `--initialize-at-random-waypoints` | disabled | Reset episodes near random waypoints instead of the start point. |
| `--initialize-at-random-gates` | disabled | Reset episodes near random gates in `figure8_gates`. |
| `--initialize-uniform` | disabled | Reset uniformly over the track area and target the nearest valid gate in `figure8_gates`. |
| `--render` | disabled | Run visualization/evaluation instead of training. |
| `--render-steps` | `2000` | Maximum steps collected per rendered attempt. |
| `--render-episodes` | `1` | Number of attempts to visualize. |
| `--simultaneous` | disabled | Show multiple rendered attempts at the same time. |
| `--record` | disabled | Save MP4 instead of only opening the viewer. |
| `--output` | `rl/runs/bebop2_waypoints_rollout.mp4` | Output video path when recording. |
| `--auto-play` / `--no-auto-play` | `--auto-play` | Start visualization playback automatically. |

Recommended next implementation steps:

1. Smoke-test [drone_ppo_bebop2.py](drone_ppo_bebop2.py) with the command above.
2. Load the trained CfC checkpoint from `checkpoints/` using the saved YAML in `configs/`.
3. Use the CfC controller as either:
   - an action prior/residual controller, where PPO learns a small correction on top of CfC commands;
   - an initialization for a compatible PyTorch policy, if the SB3 policy architecture is adapted to the CfC network.
4. Keep Bebop1 dataset normalization fixed for the CfC input path, but evaluate rewards and termination with Bebop2 simulated states.
5. Save RL outputs under this folder:
   - `configs/` for PPO/env YAMLs;
   - `envs/` for Gymnasium environments;
   - `checkpoints/` for SB3 `.zip` policies;
   - `runs/` for TensorBoard/log outputs.

For this project, the residual-controller route is usually safer: it preserves the learned Bebop1 behavior and lets PPO compensate for the Bebop2 dynamics mismatch.
