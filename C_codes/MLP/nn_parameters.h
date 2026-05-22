/*
File: nn_parameters.h
Generated from: mlp_epoch=19_val_loss=0.003130.ckpt
Model kind: mlp
Input order: dx, dy, dz, vx, vy, vz, phi, theta, psi, p, q, r, Mx_ext, My_ext, Mz_ext, omega1, omega2, omega3, omega4
*/
#ifndef NN_PARAMETERS_H
#define NN_PARAMETERS_H

#define NUM_STATES 19
#define NUM_CONTROLS 4
#define CONV_FEATURES 256
#define HIDDEN_SIZE 64
#define MLP_NODES 120


extern const float input_norm_min[19];
extern const float input_norm_max[19];
extern const float network_0_weight[2280];
extern const float network_0_bias[120];
extern const float network_2_weight[14400];
extern const float network_2_bias[120];
extern const float network_4_weight[14400];
extern const float network_4_bias[120];
extern const float network_6_weight[480];
extern const float network_6_bias[4];

#endif
