#include <stdio.h>
#include "nn_cfc_operations.h"

int main(void) {
    float state[NUM_STATES] = {0};
    float control[NUM_CONTROLS];
    nn_cfc_reset();
    nn_cfc_control(state, control);
    for (int i = 0; i < NUM_CONTROLS; ++i) printf("u%d = %.9f\n", i + 1, control[i]);
    return 0;
}
