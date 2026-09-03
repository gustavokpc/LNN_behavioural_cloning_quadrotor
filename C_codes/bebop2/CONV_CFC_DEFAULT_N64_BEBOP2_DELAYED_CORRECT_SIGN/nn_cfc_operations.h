#ifndef NN_OPERATIONS_H
#define NN_OPERATIONS_H

#include "nn_cfc_parameters.h"

/* This checkpoint reproduces ncps.CfC's implicit elapsed_time=1.0. */
#define NN_CFC_SUPPORTS_RUNTIME_TIMESPAN 0

void nn_reset(void);
void nn_control(const float *state, float *control);

#endif
