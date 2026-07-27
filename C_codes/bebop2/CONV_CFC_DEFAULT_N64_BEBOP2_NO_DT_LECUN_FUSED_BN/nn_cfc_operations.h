#ifndef NN_OPERATIONS_H
#define NN_OPERATIONS_H

#include "nn_cfc_parameters.h"

void nn_reset(void);
void nn_control(const float *state, float *control);

#endif
