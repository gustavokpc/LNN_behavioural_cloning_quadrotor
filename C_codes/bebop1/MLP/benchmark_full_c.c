/*
File: benchmark_full_c.c
Goal: Full C benchmark: controller + quadrotor dynamics + integration loop.
*/

#define _POSIX_C_SOURCE 199309L

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nn_operations.h"

#define FULL_STATE_SIZE 19

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const float G = 9.81f;
static const float IXX = 0.000906f;
static const float IYY = 0.001242f;
static const float IZZ = 0.002054f;
static const float KX = 1.07933887e-05f;
static const float KY = 9.65250793e-06f;
static const float KZ = 2.7862899e-05f;
static const float KOMEGA = 4.36301076e-08f;
static const float KH = 0.06255013f;
static const float KP = 1.4119331e-09f;
static const float KPV = -0.00797102f;
static const float KQ = 1.21601884e-09f;
static const float KQV = 0.01292637f;
static const float KR1 = 2.57035545e-06f;
static const float KR2 = 4.10923364e-07f;
static const float KRR = 0.00081293f;
static const float OMEGA_MAX = 10000.0f;
static const float OMEGA_MIN = 5000.0f;
static const float TAU = 0.06f;

static uint32_t rng_state = 42u;

static float uniform01(void)
{
    rng_state = 1664525u * rng_state + 1013904223u;
    return (float)((rng_state >> 8) & 0x00FFFFFFu) / 16777215.0f;
}

static float uniform_range(float lo, float hi)
{
    return lo + (hi - lo) * uniform01();
}

static float choice_sign(void)
{
    return uniform01() < 0.5f ? -1.0f : 1.0f;
}

static void generate_start(float *state)
{
    for (int i = 0; i < FULL_STATE_SIZE; ++i) {
        state[i] = 0.0f;
    }
    state[0] = choice_sign() * uniform_range(1.0f, 5.0f);
    state[1] = choice_sign() * uniform_range(1.0f, 5.0f);
    state[2] = uniform_range(-1.0f, 1.0f);
    state[3] = uniform_range(-0.5f, 0.5f);
    state[4] = uniform_range(-0.5f, 0.5f);
    state[5] = uniform_range(-0.5f, 0.5f);
    state[6] = uniform_range(-2.0f * (float)M_PI / 9.0f, 2.0f * (float)M_PI / 9.0f);
    state[7] = uniform_range(-2.0f * (float)M_PI / 9.0f, 2.0f * (float)M_PI / 9.0f);
    state[8] = uniform_range(-(float)M_PI, (float)M_PI);
    state[9] = uniform_range(-1.0f, 1.0f);
    state[10] = uniform_range(-1.0f, 1.0f);
    state[11] = uniform_range(-1.0f, 1.0f);
    state[12] = uniform_range(-0.01f, 0.01f);
    state[13] = uniform_range(-0.01f, 0.01f);
    state[14] = uniform_range(-0.01f, 0.01f);
    for (int i = 15; i < 19; ++i) {
        state[i] = 0.5f * (OMEGA_MAX + OMEGA_MIN);
    }
}

static void dynamics(const float *state, const float *action, float *deriv)
{
    const float dx = state[0], dy = state[1], dz = state[2];
    const float vx = state[3], vy = state[4], vz = state[5];
    const float phi = state[6], theta = state[7];
    const float p = state[9], q = state[10], r = state[11];
    const float mx = state[12], my = state[13], mz = state[14];
    const float omega1 = state[15], omega2 = state[16], omega3 = state[17], omega4 = state[18];
    const float u1 = action[0], u2 = action[1], u3 = action[2], u4 = action[3];

    const float omegas = omega1 + omega2 + omega3 + omega4;
    const float omegas2 = omega1 * omega1 + omega2 * omega2 + omega3 * omega3 + omega4 * omega4;

    const float d_omega1 = (OMEGA_MIN + u1 * (OMEGA_MAX - OMEGA_MIN) - omega1) / TAU;
    const float d_omega2 = (OMEGA_MIN + u2 * (OMEGA_MAX - OMEGA_MIN) - omega2) / TAU;
    const float d_omega3 = (OMEGA_MIN + u3 * (OMEGA_MAX - OMEGA_MIN) - omega3) / TAU;
    const float d_omega4 = (OMEGA_MIN + u4 * (OMEGA_MAX - OMEGA_MIN) - omega4) / TAU;

    const float tau_x = KP * (omega1 * omega1 - omega2 * omega2 - omega3 * omega3 + omega4 * omega4) + KPV * vy + mx;
    const float tau_y = KQ * (omega1 * omega1 + omega2 * omega2 - omega3 * omega3 - omega4 * omega4) + KQV * vx + my;
    const float tau_z = KR1 * (-omega1 + omega2 - omega3 + omega4) + KR2 * (-d_omega1 + d_omega2 - d_omega3 + d_omega4) - KRR * r + mz;

    deriv[0] = -q * dz + r * dy - vx;
    deriv[1] = p * dz - r * dx - vy;
    deriv[2] = -p * dy + q * dx - vz;
    deriv[3] = -q * vz + r * vy - G * sinf(theta) - KX * omegas * vx;
    deriv[4] = p * vz - r * vx + G * cosf(theta) * sinf(phi) - KY * omegas * vy;
    deriv[5] = -p * vy + q * vx + G * cosf(theta) * cosf(phi) - KZ * omegas * vz - KOMEGA * omegas2 - KH * (vx * vx + vy * vy);
    deriv[6] = p + q * sinf(phi) * tanf(theta) + r * cosf(phi) * tanf(theta);
    deriv[7] = q * cosf(phi) - r * sinf(phi);
    deriv[8] = q * sinf(phi) / cosf(theta) + r * cosf(phi) / cosf(theta);
    deriv[9] = (q * r * (IYY - IZZ) + tau_x) / IXX;
    deriv[10] = (p * r * (IZZ - IXX) + tau_y) / IYY;
    deriv[11] = (p * q * (IXX - IYY) + tau_z) / IZZ;
    deriv[12] = 0.0f;
    deriv[13] = 0.0f;
    deriv[14] = 0.0f;
    deriv[15] = d_omega1;
    deriv[16] = d_omega2;
    deriv[17] = d_omega3;
    deriv[18] = d_omega4;
}

static void rk4_step(float *state, const float *action, float dt)
{
    float k1[FULL_STATE_SIZE], k2[FULL_STATE_SIZE], k3[FULL_STATE_SIZE], k4[FULL_STATE_SIZE], tmp[FULL_STATE_SIZE];
    dynamics(state, action, k1);
    for (int i = 0; i < FULL_STATE_SIZE; ++i) tmp[i] = state[i] + 0.5f * dt * k1[i];
    dynamics(tmp, action, k2);
    for (int i = 0; i < FULL_STATE_SIZE; ++i) tmp[i] = state[i] + 0.5f * dt * k2[i];
    dynamics(tmp, action, k3);
    for (int i = 0; i < FULL_STATE_SIZE; ++i) tmp[i] = state[i] + dt * k3[i];
    dynamics(tmp, action, k4);
    for (int i = 0; i < FULL_STATE_SIZE; ++i) {
        state[i] += (dt / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    }
}

static double elapsed_seconds(struct timespec start, struct timespec end)
{
    return (double)(end.tv_sec - start.tv_sec) + 1.0e-9 * (double)(end.tv_nsec - start.tv_nsec);
}

int main(int argc, char **argv)
{
    const int runs = argc > 1 ? atoi(argv[1]) : 1000;
    const int horizon_steps = argc > 2 ? atoi(argv[2]) : 400;
    const float dt = argc > 3 ? strtof(argv[3], NULL) : 0.01f;
    if (runs <= 0 || horizon_steps <= 0 || dt <= 0.0f) {
        fprintf(stderr, "Usage: %s [runs=1000] [horizon_steps=400] [dt=0.01]\n", argv[0]);
        return 1;
    }

    long long total_steps = 0;
    double checksum = 0.0;
    struct timespec start_time, end_time;

    clock_gettime(CLOCK_MONOTONIC, &start_time);
    for (int run = 0; run < runs; ++run) {
        float state[FULL_STATE_SIZE];
        float control[NUM_CONTROLS];
        generate_start(state);
        nn_reset();
        for (int step = 0; step < horizon_steps; ++step) {
            nn_control(state, control);
            rk4_step(state, control, dt);
            total_steps += 1;
        }
        checksum += state[0] + state[1] + state[2] + control[0] + control[1] + control[2] + control[3];
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    const double elapsed = elapsed_seconds(start_time, end_time);
    printf("C full benchmark\n");
    printf("Runs: %d\n", runs);
    printf("Horizon steps: %d\n", horizon_steps);
    printf("Total simulated steps: %lld\n", total_steps);
    printf("Total time: %.6f s\n", elapsed);
    printf("Average rollout: %.9f s\n", elapsed / (double)runs);
    printf("Average step: %.9f s\n", elapsed / (double)total_steps);
    printf("Checksum: %.9f\n", checksum);
    return 0;
}
