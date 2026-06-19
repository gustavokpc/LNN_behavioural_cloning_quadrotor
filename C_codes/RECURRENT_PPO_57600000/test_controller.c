#include <stdio.h>
#include "nn_operations.h"

int main(void) {
    float state[NUM_STATES] = {0.3f,-0.2f,0.1f,0.05f,-0.03f,0.02f,0.01f,-0.02f,0.03f,0.1f,-0.1f,0.05f,0.0f,0.0f,0.0f,5500.0f,5600.0f,5700.0f,5800.0f};
    float control[NUM_CONTROLS];
    nn_reset();
    nn_control(state, control);
    for (int i = 0; i < NUM_CONTROLS; ++i) printf("u%d = %.9f\n", i + 1, control[i]);
    return 0;
}
