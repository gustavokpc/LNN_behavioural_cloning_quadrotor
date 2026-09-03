# LNN Behavioural Cloning Quadrotor

Treinamento supervisionado (SL), simulação e exportação de controladores neurais para Bebop1 e Bebop2.

Os comandos abaixo partem da pasta `LNN_estag`:

```bash
cd /home/gustavokpc/Documents/ESTAG/LNN_estag
```

## Estrutura principal

- `train.py` e `test.py`: treino e teste supervisionado.
- `configs/`: configurações YAML associadas aos modelos.
- `checkpoints/`: checkpoints PyTorch de SL.
- `simulators/`: simulações em malha fechada.
- `rl/`: PPO, Recurrent PPO/CfC e Recurrent PPO/LTC.
- `C_codes/`: modelos exportados para C.
- `organized_plots/`: plots, CSVs, vídeos e logs organizados.

## SL: treino e teste

Edite `train_config.yaml` e execute:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/train.py
```

O treino grava o checkpoint em `checkpoints/` e uma cópia da configuração em `configs/`.

Para testar o modelo indicado em `test_config.yaml`:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/test.py
```

Com plots:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/test.py --plot
```

## SL: simulação recente com Conv-LTC

Exemplo baseado nos últimos testes do Conv-LTC Bebop2, com `dt = 0.01 s`, `tau = 0.06 s` e dinâmica MATLAB:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python \
  LNN_behavioural_cloning_quadrotor/simulators/Simulator_gazebo_square.py \
  --checkpoint LNN_behavioural_cloning_quadrotor/checkpoints/bebop2/conv_ltc_n64_bebop2_baseline_no_dt.ckpt \
  --model-config LNN_behavioural_cloning_quadrotor/configs/bebop2/conv_ltc_n64_bebop2_baseline_no_dt.yaml \
  --normalization-limits bebop2_tau_0.06 \
  --dynamics-model quadrotor_sim_matlab \
  --rotor-yaw-sign +1 \
  --dt 0.01 \
  --time-simulation 20 \
  --dist-error 0.001 \
  --tau 0.06 \
  --device cpu \
  --plot-actions \
  --plot-signals
```

Preset models available through `--model`:

| `--model` | Config | C export |
| --- | --- | --- |
| `CFC` | `configs/bebop1_conv_cfc_h=64_seq=1_epoch=18_val_loss=0.000142.yaml` | `C_codes/CFC` |
| `CFC_PURE` | `configs/bebop1_conv_cfc_pure_h=64_seq=1_epoch=17_val_loss=0.000203.yaml` | `C_codes/CFC_PURE` |
| `CONV_CFC_DEFAULT` | `configs/bebop1_conv_cfc_h=64_seq=1_with_dt_epoch=17_val_loss=0.000326.yaml` | `C_codes/CONV_CFC_DEFAULT` |
| `NCP_CFC` | `configs/bebop1_conv_ncp_cfc_h=60_seq=1_epoch=18_val_loss=0.000143.yaml` | `C_codes/NCP_CFC` |
| `LTC` | `configs/bebop1_conv_ltc_h=64_seq=1_epoch=18_val_loss=0.000193.yaml` | `C_codes/LTC` |
| `CTRNN` | `configs/bebop1_conv_ctrnn_h=64_seq=1_epoch=19_val_loss=0.000150.yaml` | `C_codes/CTRNN` |
| `RNN` | `configs/bebop1_conv_rnn_h=64_seq=1_epoch=17_val_loss=0.000147.yaml` | `C_codes/RNN` |
| `GRU` | `configs/bebop1_conv_gru_h=64_seq=1_epoch=19_val_loss=0.000088.yaml` | `C_codes/GRU` |
| `LSTM` | `configs/bebop1_conv_lstm_h=64_seq=1_epoch=17_val_loss=0.000092.yaml` | `C_codes/LSTM` |
| `MLP` | `configs/bebop1_mlp_seq=1_epoch=19_val_loss=0.003130.yaml` | `C_codes/MLP` |

For example, `--model CFC` launches the Bebop1 CfC export in `C_codes/CFC`; `--model GRU` launches the Bebop1 GRU export; and so on. These presets are convenient when the config and C folder are already listed above.

- `--dt 0.01`: 100 Hz simulados.
- `--dt 0.02`: 50 Hz simulados.
- `--dt 0.04`: 25 Hz simulados.

`1/dt` é a frequência solicitada na simulação. A frequência efetiva em hardware depende do tempo real de inferência e do restante do loop de controle.

## SL: simulação recente com Conv-CfC

Exemplo do Bebop2 treinado com ruído e `tau = 0.01 s`:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python \
  LNN_behavioural_cloning_quadrotor/simulators/Simulator_gazebo_square.py \
  --checkpoint LNN_behavioural_cloning_quadrotor/checkpoints/bebop2_tau_001/conv_cfc_default_n64_bebop2_tau_001_double_noise_all.ckpt \
  --model-config LNN_behavioural_cloning_quadrotor/configs/bebop2_tau_001/conv_cfc_default_n64_bebop2_tau_001_double_noise_all.yaml \
  --normalization-limits bebop2_tau_0.01 \
  --dynamics-model quadrotor_sim_matlab \
  --rotor-yaw-sign +1 \
  --dt 0.01 \
  --time-simulation 20 \
  --dist-error 0.001 \
  --tau 0.01 \
  --device cpu \
  --plot-actions \
  --plot-signals
```

Para adicionar o ruído usado nos testes:

```text
--input-noise-p-sigma 0.24
--input-noise-q-sigma 0.12
--input-noise-r-sigma 0.10
```

Sem caminhos explícitos, os plots novos são gravados em:

```text
organized_plots/sl_runs/generated/python_controller/
```

Use `--action-plot-output` e `--signals-plot-output` quando quiser nomes específicos.

## Simulação com controlador C

```bash
.venv/bin/python -m \
  LNN_behavioural_cloning_quadrotor.simulators.Simulator_gazebo_square_C \
  --model CFC \
  --dynamics-model quadrotor_sim_matlab \
  --dt 0.01 \
  --tau 0.03 \
  --time-simulation 20 \
  --plot-actions \
  --plot-signals
```

Os resultados ficam em `organized_plots/sl_runs/generated/c_controller/`.

## RL

Os comandos de treino, continuação, renderização e gravação estão em [rl/README.md](rl/README.md).

## Resultados

Abra [organized_plots/index.html](organized_plots/index.html) para navegar pelos resultados.

- Experimentos SL: `organized_plots/sl_runs/experiments/`.
- Novas simulações SL: `organized_plots/sl_runs/generated/`.
- Resultados RL: `organized_plots/rl_runs/`.

Para ver todas as opções de um script:

```bash
.venv/bin/python LNN_behavioural_cloning_quadrotor/simulators/Simulator_gazebo_square.py --help
.venv/bin/python -m LNN_behavioural_cloning_quadrotor.rl.drone_ppo_bebop2 --help
```
