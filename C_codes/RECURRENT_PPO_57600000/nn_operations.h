#ifndef NN_OPERATIONS_H
#define NN_OPERATIONS_H

#include <math.h>
#include "nn_parameters.h"

void nn_reset(void);
void nn_set_hidden(const float *hidden_state);
void nn_get_hidden(float *hidden_state);
void nn_control(const float *state, float *control);
void nn_control_with_state(const float *state, float *hidden_state, float *control);

#endif
