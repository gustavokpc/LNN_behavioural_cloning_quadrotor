#ifndef NN_OPERATIONS_H
#define NN_OPERATIONS_H

#include "nn_cfc_parameters.h"

/* Nominal training/controller period, used only when no valid measured dt exists. */
#define NN_CFC_DEFAULT_TIMESPAN_S 0.01f
#define NN_CFC_SUPPORTS_RUNTIME_TIMESPAN 1

void nn_reset(void);
void nn_set_timespan(float timespan_s);
void nn_control(const float *state, float *control);

#endif
