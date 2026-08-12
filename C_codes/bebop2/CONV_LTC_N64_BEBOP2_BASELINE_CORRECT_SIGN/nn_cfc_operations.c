#include "nn_cfc_operations.h"

#include <math.h>

#define BN_EPS 1.0e-5f
#define LTC_EPS 1.0e-8f

#if NUM_STATES != 19
#error "The corrected-sign Bebop2 Conv-LTC export expects 19 physical inputs."
#endif

static float nn_ltc_state[HIDDEN_SIZE];

#if LTC_USES_RUNTIME_TIMESPAN
static float nn_ltc_timespan_s = LTC_DEFAULT_TIMESPAN;

void nn_set_timespan(float timespan_s)
{
  if (isfinite(timespan_s) && timespan_s > 0.0f) {
    nn_ltc_timespan_s = timespan_s;
  }
}
#endif

static inline float sigmoidf_local(float x)
{
  if (x >= 0.0f) {
    const float z = expf(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = expf(x);
  return z / (1.0f + z);
}

static inline float reluf_local(float x)
{
  return x > 0.0f ? x : 0.0f;
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

static void conv_features(const float *x, float *features)
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
    features[i] = 0.5f * (c4[i * 2] + c4[i * 2 + 1]);
  }
}

void nn_reset(void)
{
  for (int i = 0; i < HIDDEN_SIZE; ++i) {
    nn_ltc_state[i] = 0.0f;
  }
#if LTC_USES_RUNTIME_TIMESPAN
  nn_ltc_timespan_s = LTC_DEFAULT_TIMESPAN;
#endif
}

void nn_control(const float *state, float *control)
{
  float normalized[NUM_STATES];
  float sensory_input[CONV_FEATURES];
  float sensory_numerator[HIDDEN_SIZE];
  float sensory_denominator[HIDDEN_SIZE];
  float cm_t[HIDDEN_SIZE];

  normalize(state, normalized);
  conv_features(normalized, sensory_input);

  for (int i = 0; i < CONV_FEATURES; ++i) {
    sensory_input[i] = sensory_input[i] * ltc_input_w[i] + ltc_input_b[i];
  }

  for (int j = 0; j < HIDDEN_SIZE; ++j) {
    sensory_numerator[j] = 0.0f;
    sensory_denominator[j] = 0.0f;
  }
  for (int i = 0; i < CONV_FEATURES; ++i) {
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
      const int index = i * HIDDEN_SIZE + j;
      const float activation = sigmoidf_local(
          ltc_sensory_sigma[index] *
          (sensory_input[i] - ltc_sensory_mu[index]));
      const float conductance = ltc_sensory_w_positive[index] * activation;
      sensory_numerator[j] += conductance * ltc_sensory_erev[index];
      sensory_denominator[j] += conductance;
    }
  }

#if LTC_USES_RUNTIME_TIMESPAN
  const float elapsed_time = nn_ltc_timespan_s;
#else
  const float elapsed_time = LTC_DEFAULT_TIMESPAN;
#endif
  for (int j = 0; j < HIDDEN_SIZE; ++j) {
    cm_t[j] = ltc_cm_positive[j] /
              (elapsed_time / (float)LTC_ODE_UNFOLDS);
  }

  for (int unfold = 0; unfold < LTC_ODE_UNFOLDS; ++unfold) {
    float numerator[HIDDEN_SIZE];
    float denominator[HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
      numerator[j] = sensory_numerator[j];
      denominator[j] = sensory_denominator[j];
    }

    for (int i = 0; i < HIDDEN_SIZE; ++i) {
      for (int j = 0; j < HIDDEN_SIZE; ++j) {
        const int index = i * HIDDEN_SIZE + j;
        const float activation = sigmoidf_local(
            ltc_sigma[index] * (nn_ltc_state[i] - ltc_mu[index]));
        const float conductance = ltc_w_positive[index] * activation;
        numerator[j] += conductance * ltc_erev[index];
        denominator[j] += conductance;
      }
    }

    for (int j = 0; j < HIDDEN_SIZE; ++j) {
      const float total_numerator =
          cm_t[j] * nn_ltc_state[j] +
          ltc_gleak_positive[j] * ltc_vleak[j] + numerator[j];
      const float total_denominator =
          cm_t[j] + ltc_gleak_positive[j] + denominator[j];
      nn_ltc_state[j] = total_numerator / (total_denominator + LTC_EPS);
    }
  }

  for (int i = 0; i < NUM_CONTROLS; ++i) {
    control[i] = nn_ltc_state[i] * ltc_output_w[i] + ltc_output_b[i];
  }
  clamp_output(control);
}
