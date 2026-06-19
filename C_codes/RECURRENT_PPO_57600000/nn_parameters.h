#ifndef NN_PARAMETERS_H
#define NN_PARAMETERS_H

/* File: nn_parameters.h
 * Generated from: recurrent_ppo_bebop2_waypoints_57600000_steps.zip
 */
#define NUM_STATES 19
#define NUM_CONTROLS 4
#define CFC_INPUT_DIM 19
#define CFC_HIDDEN_DIM 64
#define CFC_BACKBONE_DIM 128
#define POLICY_HIDDEN_DIM 64
#define CFC_TIMESPAN 0.01f

extern const float OBS_MIN[19];
extern const float OBS_MAX[19];
extern const float CFC_BACKBONE0_WEIGHT[10624];
extern const float CFC_BACKBONE0_BIAS[128];
extern const float CFC_FF1_WEIGHT[8192];
extern const float CFC_FF1_BIAS[64];
extern const float CFC_FF2_WEIGHT[8192];
extern const float CFC_FF2_BIAS[64];
extern const float CFC_TIME_A_WEIGHT[8192];
extern const float CFC_TIME_A_BIAS[64];
extern const float CFC_TIME_B_WEIGHT[8192];
extern const float CFC_TIME_B_BIAS[64];
extern const float POLICY0_WEIGHT[4096];
extern const float POLICY0_BIAS[64];
extern const float POLICY2_WEIGHT[4096];
extern const float POLICY2_BIAS[64];
extern const float ACTION_WEIGHT[256];
extern const float ACTION_BIAS[4];

#endif
