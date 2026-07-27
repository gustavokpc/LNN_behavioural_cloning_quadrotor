#ifndef RL_CFC_OPERATIONS_H
#define RL_CFC_OPERATIONS_H

#include <math.h>
#include "rl_cfc_parameters.h"

void rl_cfc_reset(void);
void rl_cfc_set_hidden(const float *hidden_state);
void rl_cfc_get_hidden(float *hidden_state);
/* Deterministic checkpoint action, clipped to the trained [0, 1] range. */
void rl_cfc_control(const float *state, float *control);
void rl_cfc_control_with_state(const float *state, float *hidden_state, float *control);
void rl_reset(void);
void rl_control(const float *state, float *control);

/* Unclipped action-head output, before the checkpoint's [0, 1] action clipping. */
extern float rl_cfc_last_raw_control[NUM_CONTROLS];

#endif
