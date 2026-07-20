/*
File: nn_parameters.h
Generated from: bbp2_conv_cfc_default_n64_epoch=19_val_loss=0.000143.ckpt
Model kind: cfc_default conv front-end
Input order: dx, dy, dz, vx, vy, vz, phi, theta, psi, p, q, r, Mx_ext, My_ext, Mz_ext, omega1, omega2, omega3, omega4
Normalization: bebop2_tau_0_06
*/
#ifndef NN_PARAMETERS_H
#define NN_PARAMETERS_H

#define NUM_STATES 19
#define NUM_CONTROLS 4
#define CONV_FEATURES 256
#define HIDDEN_SIZE 64

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
extern const float rnn_rnn_cell_backbone_0_weight[40960];
extern const float rnn_rnn_cell_backbone_0_bias[128];
extern const float rnn_rnn_cell_ff1_weight[8192];
extern const float rnn_rnn_cell_ff1_bias[64];
extern const float rnn_rnn_cell_ff2_weight[8192];
extern const float rnn_rnn_cell_ff2_bias[64];
extern const float rnn_rnn_cell_time_a_weight[8192];
extern const float rnn_rnn_cell_time_a_bias[64];
extern const float rnn_rnn_cell_time_b_weight[8192];
extern const float rnn_rnn_cell_time_b_bias[64];
extern const float rnn_fc_weight[256];
extern const float rnn_fc_bias[4];

#endif
