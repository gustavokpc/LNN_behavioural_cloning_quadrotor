#include <stdio.h>
#include <stdlib.h>
#include "nn_operations.h"
int main(int argc, char **argv) {
    if (argc != NUM_STATES + 1) { fprintf(stderr, "Expected 19 state values.\n"); return 1; }
    float state[NUM_STATES], control[NUM_CONTROLS];
    for (int i=0;i<NUM_STATES;++i) state[i]=strtof(argv[i+1], NULL);
    nn_reset();
    nn_control(state, control);
    for (int i=0;i<NUM_CONTROLS;++i) printf("%.9f%s", control[i], i==NUM_CONTROLS-1 ? "\n" : " ");
    return 0;
}
