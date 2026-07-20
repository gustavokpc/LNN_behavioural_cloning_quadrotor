# Simulator Run Images

Image names use this pattern when possible:

`model__track__dyn-<dynamics>__tau-<seconds>__dist-<value>__noise-<type>__signals-<scope>.png`

Common fields:

- `model`: controller or training family, for example `cfc`, `bbp2_conv_cfc`, `correctnorm_bbp2_newnorm`.
- `track`: trajectory type, usually `square`.
- `dyn`: plant dynamics used in the simulation, for example `original` or `matlab`.
- `tau`: motor time constant used in seconds.
- `dist`: disturbance value used in the run.
- `noise`: noise condition, for example `none` or `pqr`.
- `signals`: plot scope, usually `all`.

Folders:

- `cfc/original`: normal CFC with original dynamics.
- `cfc/matlab`: normal CFC with Matlab/Bebop2-style dynamics.
- `bbp2_conv_cfc`: Bebop2 Conv-CfC experiments.
- `bbp2_correctnorm`: CorrectNorm Bebop2 experiments.
- `bbp2_newdataset`: models trained/evaluated with newer Bebop2 datasets or normalization variants.
- `noise_models`: models trained with noisy datasets or no-dt variants.
- `metrics`: aggregate metric plots.
