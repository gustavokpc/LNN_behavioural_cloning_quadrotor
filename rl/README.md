# Reinforcement Learning

Treino e avaliação de controladores PPO para o Bebop2 usando as dinâmicas do projeto.

Execute os comandos a partir de `LNN_estag`:

```bash
cd /home/gustavokpc/Documents/ESTAG/LNN_estag
```

## Políticas disponíveis

| Política | Uso |
| --- | --- |
| `ppo` | PPO com política MLP. |
| `recurrent_ppo` | Recurrent PPO com célula CfC. |
| `recurrent_ppo_ltc` | Recurrent PPO com célula LTC. |
| `bc_ppo` | Inicializa o ator PPO a partir de um checkpoint SL. |
| `residual_ppo` | Mantém o controlador SL congelado e aprende uma correção PPO. |

Tracks disponíveis:

- `square_waypoints`: trajetória quadrada; aceita todas as políticas.
- `figure8_gates`: pista em oito; use `ppo`, `recurrent_ppo` ou `recurrent_ppo_ltc`.

## Treino rápido

Recurrent PPO/CfC na pista em oito:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --track figure8_gates \
  --policy-type recurrent_ppo \
  --num-envs 16 \
  --rollout-fragment-length 512 \
  --batch-size 1024 \
  --total-timesteps 100000000 \
  --checkpoint-freq 100000 \
  --dt 0.01 \
  --motor-tau 0.025 \
  --rotor-yaw-sign +1 \
  --device auto
```

Para treinar a versão LTC, troque somente a política:

```text
--policy-type recurrent_ppo_ltc
```

Because `figure8_gates` uses the legacy-style observation vector instead of the 19-value supervised-learning input, use `--policy-type ppo`, `--policy-type recurrent_ppo`, `--policy-type recurrent_ppo_ltc`, or `--policy-type recurrent_ppo_ncp_cfc`. The `bc_ppo` and `residual_ppo` modes remain available for the waypoint environment, but are intentionally rejected for `figure8_gates`.

The `recurrent_ppo_ncp_cfc` policy uses a sparsely wired CfC. Its wiring can be
configured with `--ncp-inter-neurons`, `--ncp-command-neurons`,
`--ncp-sensory-fanout`, `--ncp-inter-fanout`,
`--ncp-recurrent-command-synapses`, and `--ncp-motor-fanin`. Their defaults are
32, 24, 20, 16, 16, and 20 respectively; `--cell-size` sets the number of NCP
motor/output neurons. `--ncp-scale-factor` scales the six wiring parameters and
defaults to 1.0.

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --policy-type bc_ppo \
  --bc-config LNN_behavioural_cloning_quadrotor/configs/bebop1_conv_cfc_h=64_seq=1_epoch=18_val_loss=0.000142.yaml \
  --bc-checkpoint bebop1_conv_cfc_h=64_seq=1_epoch=18_val_loss=0.000142.ckpt \
  --num-envs 32 \
  --seed 0 \
  --learning-rate 0.0003 \
  --rollout-fragment-length 512 \
  --batch-size 1024 \
  --total-timesteps 100000000 \
  --checkpoint-freq 100000 \
  --dt 0.01 \
  --device auto
```

Smoke test curto:

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

## Avaliação recente: Recurrent PPO/CfC

Exemplo com o checkpoint de 96 milhões de passos, `tau = 0.025 s` e sem ruído nas observações:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m \
  LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --track figure8_gates \
  --policy-type recurrent_ppo \
  --cont LNN_behavioural_cloning_quadrotor/rl/checkpoints/CFC/cfc_tau_0025/recurrent_ppo_figure8_gates_cfc_tau_0025_noise_96000000_steps.zip \
  --render \
  --dt 0.01 \
  --motor-tau 0.025 \
  --rotor-yaw-sign +1 \
  --obs-rate-noise-std 0.0 0.0 0.0 \
  --figure8-action-range 0_1 \
  --figure8-start-pos-enu 0.0 0.0 1.0 \
  --figure8-start-gate 2 \
  --plot-actions \
  --plot-signals \
  --device cpu
```

Teste com mudança de dinâmica e ruído, mantendo o mesmo checkpoint:

```text
--motor-tau 0.03
--obs-rate-noise-std 0.24 0.12 0.10
```

## Avaliação recente: Recurrent PPO/LTC

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m \
  LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --track figure8_gates \
  --policy-type recurrent_ppo_ltc \
  --cont LNN_behavioural_cloning_quadrotor/rl/checkpoints/LTC/recurrent_ppo_ltc/recurrent_ppo_ltc_figure8_gates_96000000_steps.zip \
  --render \
  --dt 0.01 \
  --motor-tau 0.06 \
  --rotor-yaw-sign +1 \
  --figure8-action-range 0_1 \
  --figure8-start-pos-enu 0.0 0.0 1.0 \
  --figure8-start-gate 2 \
  --plot-actions \
  --plot-signals \
  --device cpu
```

## Gravar vídeo

Adicione ao comando de avaliação:

```text
--bc-config LNN_behavioural_cloning_quadrotor/configs/bebop1_conv_cfc_h=64_seq=1_epoch=18_val_loss=0.000142.yaml
--bc-checkpoint bebop1_conv_cfc_h=64_seq=1_epoch=18_val_loss=0.000142.ckpt
```

Visualize a trained checkpoint:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --render \
  --cont LNN_behavioural_cloning_quadrotor/rl/checkpoints/bebop2_waypoints/ppo/ppo_bebop2_waypoints.zip \
  --policy-type ppo
```

O vídeo, os plots e os CSVs são gravados automaticamente em:

```text
organized_plots/rl_runs/videos/<track>/
organized_plots/rl_runs/action_plots/<track>/<policy>/
organized_plots/rl_runs/signal_plots/<track>/<policy>/
```

Use `--output`, `--action-plot-output` ou `--signals-plot-output` somente quando quiser escolher outro nome.

## Hot-start a partir de SL

PPO inicializado com o controlador supervisionado:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --track square_waypoints \
  --policy-type bc_ppo \
  --num-envs 32 \
  --learning-rate 1e-5 \
  --total-timesteps 5000000
```

PPO residual em torno do controlador supervisionado congelado:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 \
  --track square_waypoints \
  --policy-type residual_ppo \
  --residual-scale 0.05 \
  --learning-rate 5e-5 \
  --num-envs 32 \
  --total-timesteps 2000000
```

## Saídas

- Checkpoints: `rl/checkpoints/`.
- TensorBoard: `organized_plots/rl_runs/tensorboard/`.
- Plots e vídeos: `organized_plots/rl_runs/`.

Ajuda completa:

```bash
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 --help
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
| `--use-flatten-features` | `True` | Feature extractor choice for `recurrent_ppo`. |
| `--bc-config` | `configs/bebop1_conv_cfc_h=64_seq=1_epoch=18_val_loss=0.000142.yaml` | YAML used to rebuild the supervised CfC for `bc_ppo`. |
| `--bc-checkpoint` | `bebop1_conv_cfc_h=64_seq=1_epoch=18_val_loss=0.000142.ckpt` | Supervised CfC checkpoint used to initialize `bc_ppo`. |
| `--bc-value-hidden-dim` | `64` | Critic hidden size used by `bc_ppo`. |
| `--residual-scale` | `0.05` | Residual correction magnitude for `residual_ppo`; the final action is `clip(CfC + residual_scale * PPO, 0, 1)`. |
| `--cont` | empty | Continue from an existing SB3 `.zip` checkpoint. |
| `--dt` | `0.01` | Unified environment timestep, controller period, and CfC/LTC timespan. Loaded recurrent checkpoints are explicitly overridden to use this value. |
| `--max-steps` | `6000` | Maximum steps per episode. |
| `--track` | `square_waypoints` | `square_waypoints` keeps the current Bebop2 waypoint trainer; `figure8_gates` uses legacy-style figure-8 gates with Bebop2 dynamics. |
| `--figure8-action-range` | `0_1` | Policy action range for `figure8_gates`; use `neg1_1` only for old checkpoints trained before the `[0, 1]` figure-8 change. |
| `--dist-error` | `0.2` | Distance threshold to mark a waypoint reached. |
| `--gate-size` | `1.5` | Gate pass/collision box size for `figure8_gates`. |
| `--gates-ahead` | `1` | Number of future gates appended to the legacy-style observation in `figure8_gates`. |
| `--integration-method` | `rk4` | Dynamics integration method. |
| `--implicit-iters` | `1` | Iterations for implicit integration. |
| `--initialize-at-random-waypoints` | disabled | Reset episodes near random waypoints instead of the start point. |
| `--initialize-at-random-gates` | disabled | Reset episodes near random gates in `figure8_gates`. |
| `--initialize-uniform` | disabled | Reset uniformly over the track area and target the nearest valid gate in `figure8_gates`. |
| `--figure8-start-pos-enu X Y Z` | disabled | Start at an exact Paparazzi/Gazebo ENU position in `figure8_gates`; for example, `1.9 1.0 1.0` for `CLIMB`. |
| `--figure8-start-gate {0..7}` | `0` | Select the first target gate: `0` is `RL_F8_1`, ..., `7` is `RL_F8_8`. |
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
