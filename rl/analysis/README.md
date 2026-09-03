# RL evaluation and analysis

This package is evaluation-only. It reuses `drone_ppo_bebop2.make_env`,
`resolve_bebop2_algorithm`, the existing SB3 checkpoint loader, and the existing
quadrotor environments. It never changes reward, dynamics, policy equations, or
training defaults.

The central collector writes `metadata.json`, `episodes.csv`, `summary.json`,
`summary.csv`, compressed `steps.npz`, and `figures/`. State arrays in
`steps.npz` use world coordinates: `[x,y,z,vx,vy,vz,phi,theta,psi,p,q,r,
Mx,My,Mz,motor1..motor4]`. Policy actions, normalized Bebop motor commands,
commanded rotor speeds, and actual motor states are separate arrays.

Run commands from the directory containing the
`LNN_behavioural_cloning_quadrotor` package. Representative commands are:

```bash
python -m LNN_behavioural_cloning_quadrotor.rl.scripts.evaluate_models evaluate \
  --checkpoint CHECKPOINT.zip --architecture auto --episodes 20 --seed 0 \
  --output rl/results/nominal

python -m LNN_behavioural_cloning_quadrotor.rl.scripts.evaluate_models aggregate \
  rl/results/seed0 rl/results/seed1 rl/results/seed2 \
  --output rl/results/across_seeds.json

python -m LNN_behavioural_cloning_quadrotor.rl.scripts.evaluate_models benchmark \
  --checkpoint CHECKPOINT.zip --output rl/results/latency --warmup 100 --iterations 2000

python -m LNN_behavioural_cloning_quadrotor.rl.scripts.evaluate_learning_curve \
  --discover-from CHECKPOINT_6400000_steps.zip --episodes 5 --output rl/results/learning

python -m LNN_behavioural_cloning_quadrotor.rl.scripts.evaluate_robustness \
  --checkpoint CHECKPOINT.zip --mode initial_condition --stress-values 1 1.5 2 3 \
  --episodes 20 --output rl/results/initial_conditions

python -m LNN_behavioural_cloning_quadrotor.rl.scripts.evaluate_robustness \
  --checkpoint CHECKPOINT.zip --mode rate_noise --stress-values 0 .05 .1 .2 \
  --episodes 20 --output rl/results/rate_noise
```

Record recurrent data explicitly (it is off by default), then run probes and
event analysis from the saved result:

```bash
python -m LNN_behavioural_cloning_quadrotor.rl.scripts.evaluate_models evaluate \
  --checkpoint CHECKPOINT.zip --episodes 20 --record-recurrent --record-internals \
  --output rl/results/recurrent
python -m LNN_behavioural_cloning_quadrotor.rl.scripts.analyze_recurrent probe \
  --result rl/results/recurrent --alpha 1
python -m LNN_behavioural_cloning_quadrotor.rl.scripts.analyze_recurrent summary \
  --result rl/results/recurrent --event gate_crossing --pre .5 --post 1
```

Dropout, perturbation/recovery, saliency, and stability are intentionally
separate from headline evaluation:

```bash
python -m LNN_behavioural_cloning_quadrotor.rl.scripts.evaluate_robustness \
  --checkpoint CHECKPOINT.zip --mode dropout_velocity --stress-values .1 .2 .5 \
  --dropout-onset 1 --episodes 10 --record-recurrent --output rl/results/dropout
python -m LNN_behavioural_cloning_quadrotor.rl.scripts.evaluate_robustness \
  --checkpoint CHECKPOINT.zip --mode perturbation --stress-values 0 .5 1 \
  --perturbation-kind velocity --perturbation-vector 1 0 0 \
  --perturbation-onset 1 --episodes 10 --output rl/results/perturbation
python -m LNN_behavioural_cloning_quadrotor.rl.scripts.analyze_recurrent recovery \
  --result rl/results/perturbation/stress_002_1
python -m LNN_behavioural_cloning_quadrotor.rl.scripts.analyze_sensitivity instantaneous \
  --result rl/results/recurrent --samples 64 --virtual-controls
python -m LNN_behavioural_cloning_quadrotor.rl.scripts.analyze_sensitivity history \
  --result rl/results/recurrent --history 50 --samples 16
python -m LNN_behavioural_cloning_quadrotor.rl.scripts.analyze_stability recurrent \
  --result rl/results/recurrent --sample-index 100
python -m LNN_behavioural_cloning_quadrotor.rl.scripts.analyze_stability closed-loop \
  --result rl/results/recurrent --sample-index 100 --epsilon 1e-4 1e-5 1e-6
```

Important boundaries:

- Full-cycle success is eight environment-reported gate crossings. Target index
  wrapping is not treated as an event by itself.
- Failure breakdown retains all simultaneously reported flags and does not infer
  a priority.
- Dropout is applied after the environment constructs (and, where applicable,
  normalizes) the observation, immediately before policy inference.
- Matched and fixed-policy-dt sweeps are distinct `dt_matched` and `dt_fixed`
  modes. Episode steps are adjusted to preserve physical duration.
- CfC interpolation gates and unwired pure-CfC/LTC time constants follow the
  installed `ncps` equations exactly. Wired NCP layer-local gates are reported as
  unsupported.
- Stability outputs are sampled local diagnostics only. Gate switches and
  terminal surfaces are rejected; history/AB2 configurations are rejected when
  their complete Markov state was not saved.
