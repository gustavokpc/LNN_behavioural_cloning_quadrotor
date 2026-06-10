# RL Workspace

This folder is reserved for PPO/SB3 experiments that fine-tune or wrap the trained CfC controller on the Bebop2 MATLAB dynamics.

Recommended first implementation steps:

1. Build a Gymnasium environment around `utils.dynamics_models.quadrotor_sim_matlab`.
2. Load the trained CfC checkpoint from `checkpoints/` using the saved YAML in `configs/`.
3. Use the CfC policy as either:
   - an action prior/residual controller, where PPO learns a small correction on top of CfC commands;
   - an initialization for a compatible PyTorch policy, if the SB3 policy architecture is adapted to the CfC network.
4. Keep Bebop1 dataset normalization fixed for the CfC input path, but evaluate rewards and termination with Bebop2 simulated states.
5. Save RL outputs under this folder:
   - `configs/` for PPO/env YAMLs;
   - `envs/` for Gymnasium environments;
   - `checkpoints/` for SB3 `.zip` policies;
   - `runs/` for TensorBoard/log outputs.

For this project, the residual-controller route is usually safer: it preserves the learned Bebop1 behavior and lets PPO compensate for the Bebop2 dynamics mismatch.
