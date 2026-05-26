/*
File: nn_parameters.h
Generated from: new_NCP_CFC_60_neurons_seq_1_epoch=18_val_loss=0.000143.ckpt
Model kind: ncp
Input order: dx, dy, dz, vx, vy, vz, phi, theta, psi, p, q, r, Mx_ext, My_ext, Mz_ext, omega1, omega2, omega3, omega4
*/
#ifndef NN_PARAMETERS_H
#define NN_PARAMETERS_H

#define NUM_STATES 19
#define NUM_CONTROLS 4
#define CONV_FEATURES 256


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
extern const float rnn_rnn_cell_layer_0_sparsity_mask[9216];
extern const float rnn_rnn_cell_layer_0_ff1_weight[9216];
extern const float rnn_rnn_cell_layer_0_ff1_bias[32];
extern const float rnn_rnn_cell_layer_0_ff2_weight[9216];
extern const float rnn_rnn_cell_layer_0_ff2_bias[32];
extern const float rnn_rnn_cell_layer_0_time_a_weight[9216];
extern const float rnn_rnn_cell_layer_0_time_a_bias[32];
extern const float rnn_rnn_cell_layer_0_time_b_weight[9216];
extern const float rnn_rnn_cell_layer_0_time_b_bias[32];
extern const float rnn_rnn_cell_layer_1_sparsity_mask[1344];
extern const float rnn_rnn_cell_layer_1_ff1_weight[1344];
extern const float rnn_rnn_cell_layer_1_ff1_bias[24];
extern const float rnn_rnn_cell_layer_1_ff2_weight[1344];
extern const float rnn_rnn_cell_layer_1_ff2_bias[24];
extern const float rnn_rnn_cell_layer_1_time_a_weight[1344];
extern const float rnn_rnn_cell_layer_1_time_a_bias[24];
extern const float rnn_rnn_cell_layer_1_time_b_weight[1344];
extern const float rnn_rnn_cell_layer_1_time_b_bias[24];
extern const float rnn_rnn_cell_layer_2_sparsity_mask[112];
extern const float rnn_rnn_cell_layer_2_ff1_weight[112];
extern const float rnn_rnn_cell_layer_2_ff1_bias[4];
extern const float rnn_rnn_cell_layer_2_ff2_weight[112];
extern const float rnn_rnn_cell_layer_2_ff2_bias[4];
extern const float rnn_rnn_cell_layer_2_time_a_weight[112];
extern const float rnn_rnn_cell_layer_2_time_a_bias[4];
extern const float rnn_rnn_cell_layer_2_time_b_weight[112];
extern const float rnn_rnn_cell_layer_2_time_b_bias[4];
extern const float rnn_fc_weight[16];
extern const float rnn_fc_bias[4];

#endif
