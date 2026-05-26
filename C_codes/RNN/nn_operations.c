
#include "nn_operations.h"

#define BN_EPS 1.0e-5f
static float nn_hidden[HIDDEN_SIZE];

static inline float reluf_local(float x) { return x > 0.0f ? x : 0.0f; }

static void clamp_output(float * restrict y) {
    for (int i = 0; i < NUM_CONTROLS; ++i) {
        if (y[i] < 0.0f) y[i] = 0.0f;
        if (y[i] > 1.0f) y[i] = 1.0f;
    }
}

static void normalize(const float * restrict x, float * restrict y) {
    for (int i = 0; i < NUM_STATES; ++i) y[i] = (x[i] - input_norm_min[i]) / (input_norm_max[i] - input_norm_min[i] + 1.0e-10f);
}

static void matvec(const float * restrict x, float * restrict y, const float * restrict w, const float * restrict b, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; ++o) {
        const float *row = &w[o * in_dim];
        float acc = b[o];
        for (int i = 0; i < in_dim; ++i) acc += row[i] * x[i];
        y[o] = acc;
    }
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

void nn_reset(void) { for (int i=0;i<HIDDEN_SIZE;++i) nn_hidden[i]=0.0f; }
void nn_control(const float *state, float *control) {
    float x[NUM_STATES], feat[CONV_FEATURES];
    normalize(state, x);
    conv_features(x, feat);

    float sum[HIDDEN_SIZE];
    for (int j=0;j<HIDDEN_SIZE;++j) {
        float acc = rnn_rnn_bias_ih_l0[j] + rnn_rnn_bias_hh_l0[j];
        for (int i=0;i<CONV_FEATURES;++i) acc += rnn_rnn_weight_ih_l0[j*CONV_FEATURES+i]*feat[i];
        for (int i=0;i<HIDDEN_SIZE;++i) acc += rnn_rnn_weight_hh_l0[j*HIDDEN_SIZE+i]*nn_hidden[i];
        sum[j] = tanhf(acc);
    }
    for (int j=0;j<HIDDEN_SIZE;++j) nn_hidden[j]=sum[j];
    matvec(nn_hidden, control, rnn_fc_weight, rnn_fc_bias, HIDDEN_SIZE, NUM_CONTROLS);

    clamp_output(control);
}
