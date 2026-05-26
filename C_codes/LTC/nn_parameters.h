/*
File: nn_parameters.h
Generated from: LTC_64_neurons_seq_1_epoch=18_val_loss=0.000193.ckpt
Model kind: ltc
Input order: dx, dy, dz, vx, vy, vz, phi, theta, psi, p, q, r, Mx_ext, My_ext, Mz_ext, omega1, omega2, omega3, omega4
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
extern const float rnn_rnn_cell_gleak[64];
extern const float rnn_rnn_cell_vleak[64];
extern const float rnn_rnn_cell_cm[64];
extern const float rnn_rnn_cell_sigma[4096];
extern const float rnn_rnn_cell_mu[4096];
extern const float rnn_rnn_cell_w[4096];
extern const float rnn_rnn_cell_erev[4096];
extern const float rnn_rnn_cell_sensory_sigma[16384];
extern const float rnn_rnn_cell_sensory_mu[16384];
extern const float rnn_rnn_cell_sensory_w[16384];
extern const float rnn_rnn_cell_sensory_erev[16384];
extern const float rnn_rnn_cell_sparsity_mask[4096];
extern const float rnn_rnn_cell_sensory_sparsity_mask[16384];
extern const float rnn_rnn_cell_input_w[256];
extern const float rnn_rnn_cell_input_b[256];
extern const float rnn_rnn_cell_output_w[4];
extern const float rnn_rnn_cell_output_b[4];

#endif
