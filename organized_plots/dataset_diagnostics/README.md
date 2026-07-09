# Diagnostics Image Index

This folder groups diagnostic plots by trajectory pair or model variant.

## matched_val_normal_1413_bebop2_7

Comparison between:

- normal validation dataset trajectory 1413
- Bebop2 validation dataset trajectory 7

Files:

- `position_pair_comparison.png`: initial/final position match check.
- `closed_loop_4s_trajectory.png`: closed-loop trajectory response for 4 seconds.
- `closed_loop_4s_motor_rpm.png`: motor RPM reference/response for 4 seconds.
- `reference_horizon_trajectory.png`: trajectory response over the reference horizon.
- `reference_horizon_motor_rpm.png`: motor RPM over the reference horizon.

## matched_test_original_245_bebop2_corrected_208

Comparison between:

- original/Bebop1 test dataset trajectory 245
- corrected Bebop2 test dataset trajectory 208

Files:

- `closed_loop_4s_trajectory.png`: closed-loop trajectory response for 4 seconds.
- `closed_loop_4s_motor_rpm.png`: motor RPM reference/response for 4 seconds.
- `closed_loop_4s_motor_rpm_timespan_dt.png`: motor RPM after passing the dataset dt as CfC timespan.
- `closed_loop_4s_state_groups.png`: grouped state comparison.
- `closed_loop_4s_all_19_states.png`: all 19 state/input channels.
- `reference_horizon_motor_rpm.png`: motor RPM over the reference horizon.
- `teacher_forced_motor_rpm_bebop1_vs_bebop2.png`: direct LNN outputs using dataset inputs step by step.

## bebop2_corrected_208_model_variants

Plots focused on trajectory 208 from the corrected Bebop2 dataset.

Files:

- `correctnorm_teacher_forced_motor_rpm.png`: CorrectNorm BBP2 direct LNN outputs.
- `newdataset_conv_cfc_closed_loop_4s_motor_rpm.png`: NEWDATASET BBP2 Conv-CfC closed-loop motor RPM for 4 seconds.
