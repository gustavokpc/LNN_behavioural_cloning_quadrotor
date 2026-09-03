#ifndef RL_CFC_PARAMETERS_H
#define RL_CFC_PARAMETERS_H

/* Exact deterministic actor export from:
 * recurrent_ppo_figure8_gates_cfc_tau_0025_noise_96000000_steps.zip
 * 20 raw observations, 64-value CfC hidden state, four [0, 1] actions,
 * and a 100 Hz recurrent timestep (CfC timespan = 0.01 s).
 */
#define NUM_STATES 20
#define NUM_CONTROLS 4
#define CFC_INPUT_DIM 20
#define HIDDEN_SIZE 64
#define CFC_BACKBONE_DIM 128
#define POLICY_HIDDEN_DIM 64
#define CFC_TIMESPAN 0.01f

extern const float CFC_BACKBONE0_WEIGHT[10752];
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
