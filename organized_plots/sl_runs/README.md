# Resultados SL selecionados

Esta pasta reúne todos os resultados de aprendizado supervisionado. Os experimentos organizados mais recentes estão em `experiments/`, e novas simulações são gravadas em `generated/`.

Quando possível, os nomes seguem o padrão:

`model__track__dyn-<dynamics>__tau-<seconds>__dist-<value>__noise-<type>__signals-<scope>.png`

Campos principais:

- `model`: controlador ou família de treino.
- `track`: trajetória, normalmente `square`.
- `dyn`: dinâmica usada, por exemplo `original` ou `matlab`.
- `tau`: constante de tempo dos motores em segundos.
- `noise`: condição de ruído.
- `signals`: grupo de sinais exibido.

Grupos mais usados:

- `cfc/`: CfC com as dinâmicas original e MATLAB.
- `bbp2_conv_cfc/`: experimentos Conv-CfC do Bebop2.
- `bbp2_tau_models/`: variações da constante de tempo dos motores.
- `noise_models/`: modelos treinados com ruído e variações sem `dt`.
- `metrics/`: métricas agregadas.

Veja também o [índice completo](index.html).
