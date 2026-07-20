
#include "nn_operations.h"

#define BN_EPS 1.0e-5f
#define LTC_EPS 1.0e-8f
#define LTC_UNFOLDS 6

static float nn_ltc_state[HIDDEN_SIZE];

static inline float sigmoidf_local(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float reluf_local(float x) { return x > 0.0f ? x : 0.0f; }
static inline float softplusf_local(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

static void clamp_output(float * restrict y) {
    for (int i = 0; i < NUM_CONTROLS; ++i) {
        if (y[i] < 0.0f) y[i] = 0.0f;
        if (y[i] > 1.0f) y[i] = 1.0f;
    }
}

static void normalize(const float * restrict x, float * restrict y) {
    for (int i = 0; i < NUM_STATES; ++i) y[i] = (x[i] - input_norm_min[i]) / (input_norm_max[i] - input_norm_min[i] + 1.0e-10f);
}


static void conv1d_relu(const float *x, float *y, const float *w, const float *b, int in_ch, int in_len, int out_ch, int out_len) {
    for (int oc = 0; oc < out_ch; ++oc) {
        for (int pos = 0; pos < out_len; ++pos) {
            float acc = b[oc];
            for (int ic = 0; ic < in_ch; ++ic) {
                for (int k = 0; k < 5; ++k) {
                    int idx = pos * 2 + k - 2;
                    if (idx >= 0 && idx < in_len) acc += w[(oc * in_ch + ic) * 5 + k] * x[ic * in_len + idx];
                }
            }
            y[oc * out_len + pos] = reluf_local(acc);
        }
    }
}

static void conv1d_bn_relu(const float *x, float *y, const float *w, const float *b, const float *bn_w, const float *bn_b, const float *mean, const float *var, int in_ch, int in_len, int out_ch, int out_len) {
    for (int oc = 0; oc < out_ch; ++oc) {
        float scale = bn_w[oc] / sqrtf(var[oc] + BN_EPS);
        for (int pos = 0; pos < out_len; ++pos) {
            float acc = b[oc];
            for (int ic = 0; ic < in_ch; ++ic) {
                for (int k = 0; k < 5; ++k) {
                    int idx = pos * 2 + k - 2;
                    if (idx >= 0 && idx < in_len) acc += w[(oc * in_ch + ic) * 5 + k] * x[ic * in_len + idx];
                }
            }
            acc = (acc - mean[oc]) * scale + bn_b[oc];
            y[oc * out_len + pos] = reluf_local(acc);
        }
    }
}

static void conv_features(const float *x, float *feat) {
    float c1[64 * 10];
    float c2[128 * 5];
    float c3[128 * 3];
    float c4[256 * 2];
    conv1d_relu(x, c1, conv_block_conv1_weight, conv_block_conv1_bias, 1, NUM_STATES, 64, 10);
    conv1d_bn_relu(c1, c2, conv_block_conv2_weight, conv_block_conv2_bias, conv_block_bn2_weight, conv_block_bn2_bias, conv_block_bn2_running_mean, conv_block_bn2_running_var, 64, 10, 128, 5);
    conv1d_relu(c2, c3, conv_block_conv3_weight, conv_block_conv3_bias, 128, 5, 128, 3);
    conv1d_bn_relu(c3, c4, conv_block_conv4_weight, conv_block_conv4_bias, conv_block_bn4_weight, conv_block_bn4_bias, conv_block_bn4_running_mean, conv_block_bn4_running_var, 128, 3, 256, 2);
    for (int i = 0; i < 256; ++i) feat[i] = 0.5f * (c4[i * 2] + c4[i * 2 + 1]);
}

void nn_reset(void) { for (int i=0;i<HIDDEN_SIZE;++i) nn_ltc_state[i]=0.0f; }
void nn_control(const float *state, float *control) {
    float x[NUM_STATES], inp[CONV_FEATURES];
    normalize(state, x);
    conv_features(x, inp);
    for (int i=0;i<CONV_FEATURES;++i) inp[i] = inp[i] * rnn_rnn_cell_input_w[i] + rnn_rnn_cell_input_b[i];
    float sensory_num[HIDDEN_SIZE], sensory_den[HIDDEN_SIZE];
    for (int j=0;j<HIDDEN_SIZE;++j) { sensory_num[j]=0.0f; sensory_den[j]=0.0f; }
    for (int i=0;i<CONV_FEATURES;++i) {
        for (int j=0;j<HIDDEN_SIZE;++j) {
            float sig=sigmoidf_local(rnn_rnn_cell_sensory_sigma[i*HIDDEN_SIZE+j]*(inp[i]-rnn_rnn_cell_sensory_mu[i*HIDDEN_SIZE+j]));
            float wa=softplusf_local(rnn_rnn_cell_sensory_w[i*HIDDEN_SIZE+j])*sig*rnn_rnn_cell_sensory_sparsity_mask[i*HIDDEN_SIZE+j];
            sensory_num[j] += wa * rnn_rnn_cell_sensory_erev[i*HIDDEN_SIZE+j];
            sensory_den[j] += wa;
        }
    }
    float cm_t[HIDDEN_SIZE];
    for (int j=0;j<HIDDEN_SIZE;++j) cm_t[j]=softplusf_local(rnn_rnn_cell_cm[j])/(1.0f/(float)LTC_UNFOLDS);
    for (int step=0; step<LTC_UNFOLDS; ++step) {
        float num[HIDDEN_SIZE], den[HIDDEN_SIZE];
        for (int j=0;j<HIDDEN_SIZE;++j) { num[j]=sensory_num[j]; den[j]=sensory_den[j]; }
        for (int i=0;i<HIDDEN_SIZE;++i) {
            for (int j=0;j<HIDDEN_SIZE;++j) {
                float sig=sigmoidf_local(rnn_rnn_cell_sigma[i*HIDDEN_SIZE+j]*(nn_ltc_state[i]-rnn_rnn_cell_mu[i*HIDDEN_SIZE+j]));
                float wa=softplusf_local(rnn_rnn_cell_w[i*HIDDEN_SIZE+j])*sig*rnn_rnn_cell_sparsity_mask[i*HIDDEN_SIZE+j];
                num[j] += wa * rnn_rnn_cell_erev[i*HIDDEN_SIZE+j];
                den[j] += wa;
            }
        }
        for (int j=0;j<HIDDEN_SIZE;++j) {
            float gleak=softplusf_local(rnn_rnn_cell_gleak[j]);
            float numerator=cm_t[j]*nn_ltc_state[j] + gleak*rnn_rnn_cell_vleak[j] + num[j];
            float denominator=cm_t[j] + gleak + den[j];
            nn_ltc_state[j]=numerator/(denominator+LTC_EPS);
        }
    }
    for (int i=0;i<NUM_CONTROLS;++i) control[i]=nn_ltc_state[i]*rnn_rnn_cell_output_w[i]+rnn_rnn_cell_output_b[i];
    clamp_output(control);
}
