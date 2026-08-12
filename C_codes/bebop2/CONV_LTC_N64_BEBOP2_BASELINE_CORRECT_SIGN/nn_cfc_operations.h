#ifndef NN_OPERATIONS_H
#define NN_OPERATIONS_H

#include "nn_cfc_parameters.h"

#define NN_CFC_SUPPORTS_RUNTIME_TIMESPAN 1

void nn_reset(void);
void nn_control(const float *state, float *control);
void nn_set_timespan(float timespan_s);

#endif
