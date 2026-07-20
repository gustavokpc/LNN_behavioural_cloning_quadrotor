#ifndef NN_CFC_OPERATIONS_H
#define NN_CFC_OPERATIONS_H

#include <math.h>
#include "nn_cfc_parameters.h"

void nn_cfc_reset(void);
void nn_cfc_set_hidden(const float *hidden_state);
void nn_cfc_get_hidden(float *hidden_state);
void nn_cfc_control(const float *state, float *control);
void nn_cfc_control_with_state(const float *state, float *hidden_state, float *control);
void nn_reset(void);
void nn_control(const float *state, float *control);
extern float nn_cfc_last_raw_control[NUM_CONTROLS];

#endif
