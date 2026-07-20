#ifndef NN_CFC_PARAMETERS_H
#define NN_CFC_PARAMETERS_H

/* File: nn_cfc_parameters.h
 * Generated from: recurrent_ppo_figure8_gates.zip
 */
#define NUM_STATES 20
#define NUM_CONTROLS 4
#define CFC_INPUT_DIM 20
#define HIDDEN_SIZE 64
#define POLICY_HIDDEN_DIM 64
#define CFC_TIMESPAN 0.01f
#define LTC_ODE_UNFOLDS 6

extern const float LTC_GLEAK[64];
extern const float LTC_VLEAK[64];
extern const float LTC_CM[64];
extern const float LTC_SIGMA[4096];
extern const float LTC_MU[4096];
extern const float LTC_W[4096];
extern const float LTC_EREV[4096];
extern const float LTC_SENSORY_SIGMA[1280];
extern const float LTC_SENSORY_MU[1280];
extern const float LTC_SENSORY_W[1280];
extern const float LTC_SENSORY_EREV[1280];
extern const float LTC_SPARSITY_MASK[4096];
extern const float LTC_SENSORY_SPARSITY_MASK[1280];
extern const float LTC_INPUT_W[20];
extern const float LTC_INPUT_B[20];
extern const float LTC_OUTPUT_W[64];
extern const float LTC_OUTPUT_B[64];
extern const float POLICY0_WEIGHT[4096];
extern const float POLICY0_BIAS[64];
extern const float POLICY2_WEIGHT[4096];
extern const float POLICY2_BIAS[64];
extern const float ACTION_WEIGHT[256];
extern const float ACTION_BIAS[4];

#endif
