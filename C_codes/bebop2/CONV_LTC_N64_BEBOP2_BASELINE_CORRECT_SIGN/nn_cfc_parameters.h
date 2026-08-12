/*
File: nn_cfc_parameters.h
Generated from: conv_ltc_n64_bebop2_baseline_correct_sign.ckpt
Model kind: Conv-LTC, fully connected, 64 hidden units, 6 ODE unfolds
Input order: dx, dy, dz, vx, vy, vz, phi, theta, psi, p, q, r, Mx_ext, My_ext, Mz_ext, omega1, omega2, omega3, omega4
Normalization: bebop2_tau_0_06
LTC timespan: runtime dt supplied by nn_cfc_control (default 0.01 s)
*/
#ifndef NN_CFC_PARAMETERS_H
#define NN_CFC_PARAMETERS_H

#define NUM_STATES 19
#define NUM_CONTROLS 4
#define CONV_FEATURES 256
#define HIDDEN_SIZE 64
#define LTC_ODE_UNFOLDS 6
#define LTC_USES_RUNTIME_TIMESPAN 1
#define LTC_DEFAULT_TIMESPAN 0.01f

extern const float input_norm_min[19];
extern const float input_norm_max[19];
extern const float conv_block_conv1_weight[320];
extern const float conv_block_conv1_bias[64];
extern const float conv_block_conv2_weight[40960];
extern const float conv_block_conv2_bias[128];
extern const float conv_block_bn2_weight[128];
extern const float conv_block_bn2_bias[128];
extern const float conv_block_bn2_running_mean[128];
extern const float conv_block_bn2_running_var[128];
extern const float conv_block_conv3_weight[81920];
extern const float conv_block_conv3_bias[128];
extern const float conv_block_conv4_weight[163840];
extern const float conv_block_conv4_bias[256];
extern const float conv_block_bn4_weight[256];
extern const float conv_block_bn4_bias[256];
extern const float conv_block_bn4_running_mean[256];
extern const float conv_block_bn4_running_var[256];
extern const float ltc_vleak[64];
extern const float ltc_sigma[4096];
extern const float ltc_mu[4096];
extern const float ltc_erev[4096];
extern const float ltc_sensory_sigma[16384];
extern const float ltc_sensory_mu[16384];
extern const float ltc_sensory_erev[16384];
extern const float ltc_input_w[256];
extern const float ltc_input_b[256];
extern const float ltc_output_w[4];
extern const float ltc_output_b[4];
extern const float ltc_gleak_positive[64];
extern const float ltc_cm_positive[64];
extern const float ltc_w_positive[4096];
extern const float ltc_sensory_w_positive[16384];

#endif
