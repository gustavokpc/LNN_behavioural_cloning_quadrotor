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
  --track figure8_gates \
  --policy-type ppo \
  --num-envs 16 \
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
--record
--render-steps 6000
--no-auto-play
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
