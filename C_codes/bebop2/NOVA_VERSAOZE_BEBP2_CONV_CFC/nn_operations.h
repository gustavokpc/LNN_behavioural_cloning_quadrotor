#ifndef NN_OPERATIONS_H
#define NN_OPERATIONS_H

#include <math.h>
#include "nn_parameters.h"

void nn_reset(void);
void nn_set_timespan(float timespan);
int nn_hidden_size(void);
void nn_get_hidden(float *hidden_out);
void nn_control(const float *state, float *control);

#endif
