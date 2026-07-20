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

static float softplus_f(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static void linear(const float *input, const float *weight, const float *bias, int in_dim, int out_dim, float *output) {
    for (int o = 0; o < out_dim; ++o) {
        float sum = bias[o];
        const float *row = weight + o * in_dim;
        for (int i = 0; i < in_dim; ++i) sum += row[i] * input[i];
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

static void ltc_step(const float *state, float *hidden_state, float *readout) {
    float inputs[CFC_INPUT_DIM];
    float sensory_num[HIDDEN_SIZE];
    float sensory_den[HIDDEN_SIZE];
    float new_hidden[HIDDEN_SIZE];

    for (int i = 0; i < CFC_INPUT_DIM; ++i) {
        inputs[i] = state[i] * LTC_INPUT_W[i] + LTC_INPUT_B[i];
    }

    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        float num = 0.0f;
        float den = 0.0f;
        for (int i = 0; i < CFC_INPUT_DIM; ++i) {
            int idx = i * HIDDEN_SIZE + j;
            float act = softplus_f(LTC_SENSORY_W[idx]) * sigmoid_f(LTC_SENSORY_SIGMA[idx] * (inputs[i] - LTC_SENSORY_MU[idx]));
            act *= LTC_SENSORY_SPARSITY_MASK[idx];
            num += act * LTC_SENSORY_EREV[idx];
            den += act;
        }
        sensory_num[j] = num;
        sensory_den[j] = den;
        new_hidden[j] = hidden_state[j];
    }

    for (int unfold = 0; unfold < LTC_ODE_UNFOLDS; ++unfold) {
        float prev[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; ++i) prev[i] = new_hidden[i];

        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            float w_num = sensory_num[j];
            float w_den = sensory_den[j];
            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                int idx = i * HIDDEN_SIZE + j;
                float act = softplus_f(LTC_W[idx]) * sigmoid_f(LTC_SIGMA[idx] * (prev[i] - LTC_MU[idx]));
                act *= LTC_SPARSITY_MASK[idx];
                w_num += act * LTC_EREV[idx];
                w_den += act;
            }
            float cm_t = softplus_f(LTC_CM[j]) / (CFC_TIMESPAN / (float)LTC_ODE_UNFOLDS);
            float gleak = softplus_f(LTC_GLEAK[j]);
            float numerator = cm_t * prev[j] + gleak * LTC_VLEAK[j] + w_num;
            float denominator = cm_t + gleak + w_den;
            new_hidden[j] = numerator / (denominator + 1.0e-8f);
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        hidden_state[i] = new_hidden[i];
        readout[i] = new_hidden[i] * LTC_OUTPUT_W[i] + LTC_OUTPUT_B[i];
    }
}

void nn_cfc_control_with_state(const float *state, float *hidden_state, float *control) {
    float readout[HIDDEN_SIZE];
    float policy0[POLICY_HIDDEN_DIM];
    float policy2[POLICY_HIDDEN_DIM];
    float raw_control[NUM_CONTROLS];

    ltc_step(state, hidden_state, readout);

    linear(readout, POLICY0_WEIGHT, POLICY0_BIAS, HIDDEN_SIZE, POLICY_HIDDEN_DIM, policy0);
    for (int i = 0; i < POLICY_HIDDEN_DIM; ++i) policy0[i] = tanhf(policy0[i]);
    linear(policy0, POLICY2_WEIGHT, POLICY2_BIAS, POLICY_HIDDEN_DIM, POLICY_HIDDEN_DIM, policy2);
    for (int i = 0; i < POLICY_HIDDEN_DIM; ++i) policy2[i] = tanhf(policy2[i]);
    linear(policy2, ACTION_WEIGHT, ACTION_BIAS, POLICY_HIDDEN_DIM, NUM_CONTROLS, raw_control);

    for (int i = 0; i < NUM_CONTROLS; ++i) {
        if (raw_control[i] < -1.0f) raw_control[i] = -1.0f;
        if (raw_control[i] > 1.0f) raw_control[i] = 1.0f;
        nn_cfc_last_raw_control[i] = raw_control[i];
        control[i] = clip01(0.5f * (raw_control[i] + 1.0f));
    }
}

void nn_cfc_control(const float *state, float *control) {
    nn_cfc_control_with_state(state, nn_cfc_hidden, control);
}

void nn_reset(void) { nn_cfc_reset(); }
void nn_control(const float *state, float *control) { nn_cfc_control(state, control); }
