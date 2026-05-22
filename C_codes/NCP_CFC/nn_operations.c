
#include "nn_operations.h"

#define BN_EPS 1.0e-5f
#define LTC_EPS 1.0e-8f
#define LTC_UNFOLDS 6

static float nn_hidden[HIDDEN_SIZE];
static float nn_cell[HIDDEN_SIZE];
static float nn_ltc_state[HIDDEN_SIZE];
static float nn_ncp_state[60];

static float sigmoidf_local(float x) { return 1.0f / (1.0f + expf(-x)); }
static float reluf_local(float x) { return x > 0.0f ? x : 0.0f; }
static float lecun_tanhf(float x) { return 1.7159f * tanhf(0.666f * x); }
static float softplusf_local(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

static void clamp_output(float *y) {
    for (int i = 0; i < NUM_CONTROLS; ++i) {
        if (y[i] < 0.0f) y[i] = 0.0f;
        if (y[i] > 1.0f) y[i] = 1.0f;
    }
}

static void normalize(const float *x, float *y) {
    for (int i = 0; i < NUM_STATES; ++i) y[i] = (x[i] - input_norm_min[i]) / (input_norm_max[i] - input_norm_min[i] + 1.0e-10f);
}

static void matvec(const float *x, float *y, const float *w, const float *b, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; ++o) {
        float acc = b ? b[o] : 0.0f;
        for (int i = 0; i < in_dim; ++i) acc += w[o * in_dim + i] * x[i];
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

static void ncp_layer(const float *inp, int in_dim, const float *hx, int h_dim, const float *mask, const float *w1, const float *b1, const float *w2, const float *b2, const float *wa, const float *ba, const float *wb, const float *bb, float *out) {
    int cat_dim = in_dim + h_dim;
    float cat[320];
    float ff1[64], ff2[64], ta[64], tb[64];
    for (int i=0;i<in_dim;++i) cat[i]=inp[i];
    for (int i=0;i<h_dim;++i) cat[in_dim+i]=hx[i];
    for (int o=0;o<h_dim;++o) {
        float a=b1[o], b=b2[o], c=ba[o], d=bb[o];
        for (int i=0;i<cat_dim;++i) { float m=mask[o*cat_dim+i]; a += w1[o*cat_dim+i]*m*cat[i]; b += w2[o*cat_dim+i]*m*cat[i]; c += wa[o*cat_dim+i]*cat[i]; d += wb[o*cat_dim+i]*cat[i]; }
        ff1[o]=tanhf(a); ff2[o]=tanhf(b); ta[o]=c; tb[o]=d;
    }
    for (int o=0;o<h_dim;++o) { float t=sigmoidf_local(ta[o]+tb[o]); out[o]=ff1[o]*(1.0f-t)+t*ff2[o]; }
}
void nn_reset(void) { for (int i=0;i<60;++i) nn_ncp_state[i]=0.0f; }
void nn_control(const float *state, float *control) {
    float x[NUM_STATES], feat[CONV_FEATURES], h0[32], h1[24], h2[4];
    normalize(state, x);
    conv_features(x, feat);
    ncp_layer(feat,256,&nn_ncp_state[0],32,rnn_rnn_cell_layer_0_sparsity_mask,rnn_rnn_cell_layer_0_ff1_weight,rnn_rnn_cell_layer_0_ff1_bias,rnn_rnn_cell_layer_0_ff2_weight,rnn_rnn_cell_layer_0_ff2_bias,rnn_rnn_cell_layer_0_time_a_weight,rnn_rnn_cell_layer_0_time_a_bias,rnn_rnn_cell_layer_0_time_b_weight,rnn_rnn_cell_layer_0_time_b_bias,h0);
    ncp_layer(h0,32,&nn_ncp_state[32],24,rnn_rnn_cell_layer_1_sparsity_mask,rnn_rnn_cell_layer_1_ff1_weight,rnn_rnn_cell_layer_1_ff1_bias,rnn_rnn_cell_layer_1_ff2_weight,rnn_rnn_cell_layer_1_ff2_bias,rnn_rnn_cell_layer_1_time_a_weight,rnn_rnn_cell_layer_1_time_a_bias,rnn_rnn_cell_layer_1_time_b_weight,rnn_rnn_cell_layer_1_time_b_bias,h1);
    ncp_layer(h1,24,&nn_ncp_state[56],4,rnn_rnn_cell_layer_2_sparsity_mask,rnn_rnn_cell_layer_2_ff1_weight,rnn_rnn_cell_layer_2_ff1_bias,rnn_rnn_cell_layer_2_ff2_weight,rnn_rnn_cell_layer_2_ff2_bias,rnn_rnn_cell_layer_2_time_a_weight,rnn_rnn_cell_layer_2_time_a_bias,rnn_rnn_cell_layer_2_time_b_weight,rnn_rnn_cell_layer_2_time_b_bias,h2);
    for (int i=0;i<32;++i) nn_ncp_state[i]=h0[i]; for (int i=0;i<24;++i) nn_ncp_state[32+i]=h1[i]; for (int i=0;i<4;++i) nn_ncp_state[56+i]=h2[i];
    matvec(h2, control, rnn_fc_weight, rnn_fc_bias, 4, NUM_CONTROLS);
    clamp_output(control);
}
