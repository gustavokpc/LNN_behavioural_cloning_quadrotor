#include "nn_cfc_operations.h"

static float nn_cfc_hidden[HIDDEN_SIZE];
float nn_cfc_last_raw_control[NUM_CONTROLS];

static float clip01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

static float sigmoid_f(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

static float lecun_tanh_f(float x) {
    return 1.7159f * tanhf(0.666f * x);
}

static void linear(const float *input, const float *weight, const float *bias,
                   int in_dim, int out_dim, float *output) {
    for (int o = 0; o < out_dim; ++o) {
        float sum = bias[o];
        const float *row = weight + o * in_dim;
        for (int i = 0; i < in_dim; ++i) {
            sum += row[i] * input[i];
        }
        output[o] = sum;
    }
}

void nn_cfc_reset(void) {
    for (int i = 0; i < HIDDEN_SIZE; ++i) nn_cfc_hidden[i] = 0.0f;
}

void nn_cfc_set_hidden(const float *hidden_state) {
    for (int i = 0; i < HIDDEN_SIZE; ++i) nn_cfc_hidden[i] = hidden_state[i];
}

void nn_cfc_get_hidden(float *hidden_state) {
    for (int i = 0; i < HIDDEN_SIZE; ++i) hidden_state[i] = nn_cfc_hidden[i];
}

void nn_cfc_control_with_state(const float *state, float *hidden_state, float *control) {
    float obs[CFC_INPUT_DIM];
    float cfc_input[CFC_INPUT_DIM + HIDDEN_SIZE];
    float backbone[CFC_BACKBONE_DIM];
    float ff1[HIDDEN_SIZE];
    float ff2[HIDDEN_SIZE];
    float time_a[HIDDEN_SIZE];
    float time_b[HIDDEN_SIZE];
    float new_hidden[HIDDEN_SIZE];
    float policy0[POLICY_HIDDEN_DIM];
    float policy2[POLICY_HIDDEN_DIM];

    for (int i = 0; i < CFC_INPUT_DIM; ++i) {
        float denom = input_norm_max[i] - input_norm_min[i] + 1.0e-10f;
        obs[i] = (state[i] - input_norm_min[i]) / denom;
        cfc_input[i] = obs[i];
    }
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        cfc_input[CFC_INPUT_DIM + i] = hidden_state[i];
    }

    linear(cfc_input, CFC_BACKBONE0_WEIGHT, CFC_BACKBONE0_BIAS,
           CFC_INPUT_DIM + HIDDEN_SIZE, CFC_BACKBONE_DIM, backbone);
    for (int i = 0; i < CFC_BACKBONE_DIM; ++i) backbone[i] = lecun_tanh_f(backbone[i]);

    linear(backbone, CFC_FF1_WEIGHT, CFC_FF1_BIAS, CFC_BACKBONE_DIM, HIDDEN_SIZE, ff1);
    linear(backbone, CFC_FF2_WEIGHT, CFC_FF2_BIAS, CFC_BACKBONE_DIM, HIDDEN_SIZE, ff2);
    linear(backbone, CFC_TIME_A_WEIGHT, CFC_TIME_A_BIAS, CFC_BACKBONE_DIM, HIDDEN_SIZE, time_a);
    linear(backbone, CFC_TIME_B_WEIGHT, CFC_TIME_B_BIAS, CFC_BACKBONE_DIM, HIDDEN_SIZE, time_b);

    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        float y1 = tanhf(ff1[i]);
        float y2 = tanhf(ff2[i]);
        float interp = sigmoid_f(time_a[i] * CFC_TIMESPAN + time_b[i]);
        new_hidden[i] = y1 * (1.0f - interp) + interp * y2;
        hidden_state[i] = new_hidden[i];
    }

    linear(hidden_state, POLICY0_WEIGHT, POLICY0_BIAS, HIDDEN_SIZE, POLICY_HIDDEN_DIM, policy0);
    for (int i = 0; i < POLICY_HIDDEN_DIM; ++i) policy0[i] = tanhf(policy0[i]);

    linear(policy0, POLICY2_WEIGHT, POLICY2_BIAS, POLICY_HIDDEN_DIM, POLICY_HIDDEN_DIM, policy2);
    for (int i = 0; i < POLICY_HIDDEN_DIM; ++i) policy2[i] = tanhf(policy2[i]);

    linear(policy2, ACTION_WEIGHT, ACTION_BIAS, POLICY_HIDDEN_DIM, NUM_CONTROLS, control);
    for (int i = 0; i < NUM_CONTROLS; ++i) {
        nn_cfc_last_raw_control[i] = control[i];
        control[i] = clip01(control[i]);
    }
}

void nn_cfc_control(const float *state, float *control) {
    nn_cfc_control_with_state(state, nn_cfc_hidden, control);
}
