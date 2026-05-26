#ifndef NN_OPERATIONS_H
#define NN_OPERATIONS_H

#include <math.h>
#include "nn_parameters.h"

void nn_reset(void);
void nn_control(const float *state, float *control);

#endif
