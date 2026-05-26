/*
File: nn_parameters.h
Generated from: new_GRU_64_neurons_seq_1_epoch=19_val_loss=0.000088.ckpt
Model kind: gru
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
extern const float rnn_gru_weight_ih_l0[49152];
extern const float rnn_gru_weight_hh_l0[12288];
extern const float rnn_gru_bias_ih_l0[192];
extern const float rnn_gru_bias_hh_l0[192];
extern const float rnn_readout_layer_weight[256];
extern const float rnn_readout_layer_bias[4];

#endif
