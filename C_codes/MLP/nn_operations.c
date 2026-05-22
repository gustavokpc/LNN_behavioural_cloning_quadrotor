#include "nn_operations.h"

static float reluf_local(float x) { return x > 0.0f ? x : 0.0f; }
static void clamp_output(float *y) { for (int i=0;i<NUM_CONTROLS;++i) { if (y[i] < 0.0f) y[i]=0.0f; if (y[i] > 1.0f) y[i]=1.0f; } }
static void normalize(const float *x, float *y) { for (int i=0;i<NUM_STATES;++i) y[i]=(x[i]-input_norm_min[i])/(input_norm_max[i]-input_norm_min[i]+1.0e-10f); }
static void matvec(const float *x, float *y, const float *w, const float *b, int in_dim, int out_dim) { for (int o=0;o<out_dim;++o) { float acc=b?b[o]:0.0f; for (int i=0;i<in_dim;++i) acc += w[o*in_dim+i]*x[i]; y[o]=acc; } }
void nn_reset(void) {}
void nn_control(const float *state, float *control) {
    float x[NUM_STATES], h1[MLP_NODES], h2[MLP_NODES], h3[MLP_NODES];
    normalize(state, x);
    matvec(x, h1, network_0_weight, network_0_bias, NUM_STATES, MLP_NODES); for (int i=0;i<MLP_NODES;++i) h1[i]=reluf_local(h1[i]);
    matvec(h1, h2, network_2_weight, network_2_bias, MLP_NODES, MLP_NODES); for (int i=0;i<MLP_NODES;++i) h2[i]=reluf_local(h2[i]);
    matvec(h2, h3, network_4_weight, network_4_bias, MLP_NODES, MLP_NODES); for (int i=0;i<MLP_NODES;++i) h3[i]=reluf_local(h3[i]);
    matvec(h3, control, network_6_weight, network_6_bias, MLP_NODES, NUM_CONTROLS);
    clamp_output(control);
}
