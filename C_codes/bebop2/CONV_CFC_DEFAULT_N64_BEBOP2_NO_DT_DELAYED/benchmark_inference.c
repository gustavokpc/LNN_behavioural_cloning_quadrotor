#define _POSIX_C_SOURCE 200809L

#include "nn_cfc_operations.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WARMUP_ITERATIONS 100U
#define MEASURED_ITERATIONS 2000U

static uint64_t monotonic_ns(void)
{
  struct timespec timestamp;
  clock_gettime(CLOCK_MONOTONIC, &timestamp);
  return (uint64_t)timestamp.tv_sec * 1000000000ULL + (uint64_t)timestamp.tv_nsec;
}

static int compare_u64(const void *lhs, const void *rhs)
{
  const uint64_t a = *(const uint64_t *)lhs;
  const uint64_t b = *(const uint64_t *)rhs;
  return (a > b) - (a < b);
}

static void update_state(float state[NUM_STATES], unsigned int iteration)
{
  const float direction = (iteration & 1U) ? 1.0f : -1.0f;
  state[0] += direction * 1.0e-5f;
  state[3] -= direction * 5.0e-6f;
  state[9] += direction * 2.0e-6f;
  state[15 + (iteration & 3U)] += direction * 1.0e-3f;
}

int main(void)
{
  float state[NUM_STATES] = {
    0.30f, -0.20f, 0.10f,
    0.05f, -0.03f, 0.02f,
    0.01f, -0.02f, 0.03f,
    0.10f, -0.10f, 0.05f,
    0.0f, 0.0f, 0.0f,
    7746.62f, 7746.62f, 7746.62f, 7746.62f
  };
  float output[NUM_CONTROLS] = {0.0f};
  uint64_t samples[MEASURED_ITERATIONS];
  long double sum_ns = 0.0L;
  volatile float checksum = 0.0f;

  nn_reset();
  for (unsigned int i = 0U; i < WARMUP_ITERATIONS; ++i) {
    update_state(state, i);
    nn_control(state, output);
    checksum += output[i & 3U];
  }

  for (unsigned int i = 0U; i < MEASURED_ITERATIONS; ++i) {
    update_state(state, i);
    const uint64_t start_ns = monotonic_ns();
    nn_control(state, output);
    const uint64_t elapsed_ns = monotonic_ns() - start_ns;
    samples[i] = elapsed_ns;
    sum_ns += (long double)elapsed_ns;
    checksum += output[i & 3U];
  }

  qsort(samples, MEASURED_ITERATIONS, sizeof(samples[0]), compare_u64);
  const uint64_t median_ns = samples[MEASURED_ITERATIONS / 2U];
  const uint64_t p95_ns = samples[(MEASURED_ITERATIONS * 95U) / 100U];
  const uint64_t p99_ns = samples[(MEASURED_ITERATIONS * 99U) / 100U];

  printf("iterations=%u\n", MEASURED_ITERATIONS);
  printf("mean_us=%.3Lf\n", sum_ns / (long double)MEASURED_ITERATIONS / 1000.0L);
  printf("min_us=%.3f\n", (double)samples[0] / 1000.0);
  printf("median_us=%.3f\n", (double)median_ns / 1000.0);
  printf("p95_us=%.3f\n", (double)p95_ns / 1000.0);
  printf("p99_us=%.3f\n", (double)p99_ns / 1000.0);
  printf("max_us=%.3f\n", (double)samples[MEASURED_ITERATIONS - 1U] / 1000.0);
  printf("checksum=%.9g\n", (double)checksum);
  return 0;
}
