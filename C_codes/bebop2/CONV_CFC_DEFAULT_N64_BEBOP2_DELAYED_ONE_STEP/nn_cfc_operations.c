#include "nn_cfc_operations.h"

#include <math.h>

#define BN_EPS 1.0e-5f

#if NUM_STATES != 19
#error "This Conv-CfC export expects exactly 19 physical state inputs."
#endif

static float nn_hidden[HIDDEN_SIZE];
static float nn_timespan_s = NN_CFC_DEFAULT_TIMESPAN_S;

static inline float sigmoidf_local(float x)
{
  return 1.0f / (1.0f + expf(-x));
}

static inline float reluf_local(float x)
{
  return x > 0.0f ? x : 0.0f;
}

static inline float lecun_tanhf_local(float x)
{
  return 1.7159f * tanhf(0.666f * x);
}

static void clamp_output(float *restrict y)
{
  for (int i = 0; i < NUM_CONTROLS; ++i) {
    if (y[i] < 0.0f) {
      y[i] = 0.0f;
    } else if (y[i] > 1.0f) {
      y[i] = 1.0f;
    }
  }
}

static void normalize(const float *restrict x, float *restrict y)
{
  for (int i = 0; i < NUM_STATES; ++i) {
    y[i] = (x[i] - input_norm_min[i]) /
           (input_norm_max[i] - input_norm_min[i] + 1.0e-10f);
  }
}

static void matvec(const float *restrict x, float *restrict y,
                   const float *restrict w, const float *restrict b,
                   int in_dim, int out_dim)
{
  for (int o = 0; o < out_dim; ++o) {
    const float *row = &w[o * in_dim];
    float acc = b[o];
    for (int i = 0; i < in_dim; ++i) {
      acc += row[i] * x[i];
    }
    y[o] = acc;
  }
}

static void conv1d_relu(const float *x, float *y,
                        const float *w, const float *b,
                        int in_ch, int in_len, int out_ch, int out_len)
{
  for (int oc = 0; oc < out_ch; ++oc) {
    for (int pos = 0; pos < out_len; ++pos) {
      float acc = b[oc];
      for (int ic = 0; ic < in_ch; ++ic) {
        for (int k = 0; k < 5; ++k) {
          const int idx = pos * 2 + k - 2;
          if (idx >= 0 && idx < in_len) {
            acc += w[(oc * in_ch + ic) * 5 + k] * x[ic * in_len + idx];
          }
        }
      }
      y[oc * out_len + pos] = reluf_local(acc);
    }
  }
}

static void conv1d_bn_relu(const float *x, float *y,
                           const float *w, const float *b,
                           const float *bn_w, const float *bn_b,
                           const float *mean, const float *var,
                           int in_ch, int in_len, int out_ch, int out_len)
{
  for (int oc = 0; oc < out_ch; ++oc) {
    const float scale = bn_w[oc] / sqrtf(var[oc] + BN_EPS);
    for (int pos = 0; pos < out_len; ++pos) {
      float acc = b[oc];
      for (int ic = 0; ic < in_ch; ++ic) {
        for (int k = 0; k < 5; ++k) {
          const int idx = pos * 2 + k - 2;
          if (idx >= 0 && idx < in_len) {
            acc += w[(oc * in_ch + ic) * 5 + k] * x[ic * in_len + idx];
          }
        }
      }
      acc = (acc - mean[oc]) * scale + bn_b[oc];
      y[oc * out_len + pos] = reluf_local(acc);
    }
  }
}

static void conv_features(const float *x, float *feat)
{
  float c1[64 * 10];
  float c2[128 * 5];
  float c3[128 * 3];
  float c4[256 * 2];

  conv1d_relu(x, c1,
              conv_block_conv1_weight, conv_block_conv1_bias,
              1, NUM_STATES, 64, 10);
  conv1d_bn_relu(c1, c2,
                 conv_block_conv2_weight, conv_block_conv2_bias,
                 conv_block_bn2_weight, conv_block_bn2_bias,
                 conv_block_bn2_running_mean, conv_block_bn2_running_var,
                 64, 10, 128, 5);
  conv1d_relu(c2, c3,
              conv_block_conv3_weight, conv_block_conv3_bias,
              128, 5, 128, 3);
  conv1d_bn_relu(c3, c4,
                 conv_block_conv4_weight, conv_block_conv4_bias,
                 conv_block_bn4_weight, conv_block_bn4_bias,
                 conv_block_bn4_running_mean, conv_block_bn4_running_var,
                 128, 3, 256, 2);

  for (int i = 0; i < CONV_FEATURES; ++i) {
    feat[i] = 0.5f * (c4[i * 2] + c4[i * 2 + 1]);
  }
}

void nn_reset(void)
{
  for (int i = 0; i < HIDDEN_SIZE; ++i) {
    nn_hidden[i] = 0.0f;
  }
  nn_timespan_s = NN_CFC_DEFAULT_TIMESPAN_S;
}

void nn_set_timespan(float timespan_s)
{
  if (isfinite(timespan_s) && timespan_s > 0.0f) {
    nn_timespan_s = timespan_s;
  } else {
    nn_timespan_s = NN_CFC_DEFAULT_TIMESPAN_S;
  }
}

void nn_control(const float *state, float *control)
{
  float x[NUM_STATES];
  float feat[CONV_FEATURES];
  float cat[CONV_FEATURES + HIDDEN_SIZE];
  float backbone[128];
  float ff1[HIDDEN_SIZE];
  float ff2[HIDDEN_SIZE];
  float time_a[HIDDEN_SIZE];
  float time_b[HIDDEN_SIZE];

  normalize(state, x);
  conv_features(x, feat);

  for (int i = 0; i < CONV_FEATURES; ++i) {
    cat[i] = feat[i];
  }
  for (int i = 0; i < HIDDEN_SIZE; ++i) {
    cat[CONV_FEATURES + i] = nn_hidden[i];
  }

  matvec(cat, backbone,
         rnn_rnn_cell_backbone_0_weight,
         rnn_rnn_cell_backbone_0_bias,
         CONV_FEATURES + HIDDEN_SIZE, 128);
  for (int i = 0; i < 128; ++i) {
    backbone[i] = lecun_tanhf_local(backbone[i]);
  }

  matvec(backbone, ff1,
         rnn_rnn_cell_ff1_weight, rnn_rnn_cell_ff1_bias,
         128, HIDDEN_SIZE);
  matvec(backbone, ff2,
         rnn_rnn_cell_ff2_weight, rnn_rnn_cell_ff2_bias,
         128, HIDDEN_SIZE);
  matvec(backbone, time_a,
         rnn_rnn_cell_time_a_weight, rnn_rnn_cell_time_a_bias,
         128, HIDDEN_SIZE);
  matvec(backbone, time_b,
         rnn_rnn_cell_time_b_weight, rnn_rnn_cell_time_b_bias,
         128, HIDDEN_SIZE);

  for (int i = 0; i < HIDDEN_SIZE; ++i) {
    const float f1 = tanhf(ff1[i]);
    const float f2 = tanhf(ff2[i]);
    const float interpolation =
        sigmoidf_local(time_a[i] * nn_timespan_s + time_b[i]);
    nn_hidden[i] = f1 * (1.0f - interpolation) + interpolation * f2;
  }

  matvec(nn_hidden, control,
         rnn_fc_weight, rnn_fc_bias,
         HIDDEN_SIZE, NUM_CONTROLS);
  clamp_output(control);
}
