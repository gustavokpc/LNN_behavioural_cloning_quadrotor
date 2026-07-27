/*
 * rl_cfc_control.c
 * Paparazzi wrapper for the recurrent PPO CfC baseline controller.
 *
 * IMPORTANT:
 * - This file builds the 20-element raw RL observation expected by
 *   rl_cfc_parameters.h:
 *   pos_G, vel_G, euler_B_to_G, rates_B, motor_state, next_gate_G.
 * - The RL policy returns normalized motor actions directly in [0,1], matching
 *   the action space used to train and save the 83,200,000-step checkpoint.
 *   The wrapper maps them directly to Bebop RPM references.
 * - The airframe command_laws keep the normal autopilot actuator path as a
 *   fallback; rl_cfc_control_apply_motor_rpm() overwrites it only when the RL
 *   controller is enabled.
 */

#include "modules/rl_cfc_control/rl_cfc_control.h"
#include "modules/rl_cfc_control/rl_cfc_operations.h"

#include "paparazzi.h"
#include "autopilot.h"
#include "state.h"
#include "generated/airframe.h"
#include "generated/flight_plan.h"
#include "generated/modules.h"
#include "mcu_periph/sys_time.h"
#include "modules/actuators/motor_mixing.h"
#include "modules/nav/waypoints.h"

#ifdef MODULE_LOGGER_FILE_ID
#include "modules/loggers/logger_file.h"
#endif

#if defined(BOARD_BEBOP)
#define RL_CFC_HAS_BEBOP_ACTUATORS 1
#include "boards/bebop/actuators.h"
#else
#define RL_CFC_HAS_BEBOP_ACTUATORS 0
#endif

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#ifndef RL_CFC_GATE_SIZE_M
#define RL_CFC_GATE_SIZE_M 1.5f
#endif

#ifndef RL_CFC_TARGET_ALT_M
#define RL_CFC_TARGET_ALT_M 1.5f
#endif

#ifndef RL_CFC_START_WAYPOINT_INDEX
#define RL_CFC_START_WAYPOINT_INDEX 2
#endif

/* RPM is the RL command scale and the Bebop BLDC command unit. */
#ifndef RL_CFC_MIN_RPM
#define RL_CFC_MIN_RPM 5000.0f
#endif

#ifndef RL_CFC_MAX_RPM
#define RL_CFC_MAX_RPM 10000.0f
#endif

#ifndef RL_CFC_TRAIN_HOVER_RPM
#define RL_CFC_TRAIN_HOVER_RPM 7746.6f
#endif

/*
 * Gazebo/NPS does not provide measured Bebop motor RPM. Model the feedback
 * used by the policy as a first-order motor response instead of feeding the
 * previous command straight back into the observation.
 */
#ifndef RL_CFC_SIM_MOTOR_TAU_S
#ifdef NPS_GAZEBO_BEBOP2_MATLAB_MOTOR_TAU_S
#define RL_CFC_SIM_MOTOR_TAU_S NPS_GAZEBO_BEBOP2_MATLAB_MOTOR_TAU_S
#else
#define RL_CFC_SIM_MOTOR_TAU_S 0.06f
#endif
#endif

#ifndef RL_CFC_USE_NED_INPUT
#define RL_CFC_USE_NED_INPUT 1
#endif

#ifndef RL_CFC_OBS_Z_SIGN
#define RL_CFC_OBS_Z_SIGN 1.0f
#endif

#ifndef RL_CFC_OBS_VZ_SIGN
#define RL_CFC_OBS_VZ_SIGN 1.0f
#endif

#ifndef RL_CFC_OBS_ROLL_SIGN
#define RL_CFC_OBS_ROLL_SIGN 1.0f
#endif

#ifndef RL_CFC_OBS_PITCH_SIGN
#define RL_CFC_OBS_PITCH_SIGN 1.0f
#endif

#ifndef RL_CFC_OBS_YAW_SIGN
#define RL_CFC_OBS_YAW_SIGN 1.0f
#endif

#ifndef RL_CFC_OBS_P_SIGN
#define RL_CFC_OBS_P_SIGN 1.0f
#endif

#ifndef RL_CFC_OBS_Q_SIGN
#define RL_CFC_OBS_Q_SIGN 1.0f
#endif

#ifndef RL_CFC_OBS_R_SIGN
#define RL_CFC_OBS_R_SIGN 1.0f
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Optional NPS-only observation noise.  The aerodynamic NPS airframe bypasses
 * AHRS/INS, so the ordinary simulated sensor noise does not reach stateGet*().
 * Injecting it here makes the values consumed by the policy noisy while
 * leaving the real AP build and its estimator completely untouched.
 */
#if defined(SITL) && defined(NPS_RL_CFC_OBS_RATE_NOISE) && NPS_RL_CFC_OBS_RATE_NOISE
#ifndef NPS_RL_CFC_OBS_RATE_NOISE_STD_P
#define NPS_RL_CFC_OBS_RATE_NOISE_STD_P 0.24f
#endif
#ifndef NPS_RL_CFC_OBS_RATE_NOISE_STD_Q
#define NPS_RL_CFC_OBS_RATE_NOISE_STD_Q 0.12f
#endif
#ifndef NPS_RL_CFC_OBS_RATE_NOISE_STD_R
#define NPS_RL_CFC_OBS_RATE_NOISE_STD_R 0.10f
#endif
#ifndef NPS_RL_CFC_OBS_NOISE_SEED
#define NPS_RL_CFC_OBS_NOISE_SEED 83200001U
#endif

static uint32_t rl_cfc_obs_noise_state = NPS_RL_CFC_OBS_NOISE_SEED;

static float rl_cfc_obs_noise_uniform(void)
{
  rl_cfc_obs_noise_state = 1664525U * rl_cfc_obs_noise_state + 1013904223U;
  return ((float)((rl_cfc_obs_noise_state >> 8) + 1U)) / 16777217.0f;
}

static float rl_cfc_obs_noise_gaussian(void)
{
  const float u1 = rl_cfc_obs_noise_uniform();
  const float u2 = rl_cfc_obs_noise_uniform();
  return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static void rl_cfc_add_nps_observation_noise(float obs[NUM_STATES])
{
  obs[9] += NPS_RL_CFC_OBS_RATE_NOISE_STD_P * rl_cfc_obs_noise_gaussian();
  obs[10] += NPS_RL_CFC_OBS_RATE_NOISE_STD_Q * rl_cfc_obs_noise_gaussian();
  obs[11] += NPS_RL_CFC_OBS_RATE_NOISE_STD_R * rl_cfc_obs_noise_gaussian();
}
#else
static void rl_cfc_add_nps_observation_noise(float obs[NUM_STATES])
{
  (void)obs;
}
#endif

typedef struct {
  float x;
  float y;
  float z;
} rl_vec3_t;

typedef struct {
  float phi;
  float theta;
  float psi;
} rl_euler_t;

bool rl_cfc_control_enabled = false;
unsigned int rl_cfc_control_waypoint_index = 0;
float rl_cfc_control_last_norm[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_last_rpm[4] = {0.f, 0.f, 0.f, 0.f};
int32_t rl_cfc_control_motor_rpm_cmd[4] = {0, 0, 0, 0};
int32_t rl_cfc_control_applied_rpm_cmd[4] = {0, 0, 0, 0};
int32_t rl_cfc_control_raw_mean_rpm = 0;
unsigned int rl_cfc_control_periodic_count = 0U;
uint32_t rl_cfc_control_periodic_dt_us = 0U;
uint32_t rl_cfc_control_sensor_read_time_us = 0U;
uint32_t rl_cfc_control_inference_time_us = 0U;
uint32_t rl_cfc_control_total_time_us = 0U;
bool rl_cfc_control_use_ned_input = RL_CFC_USE_NED_INPUT;
float rl_cfc_control_target[3] = {0.f, 0.f, 0.f};
float rl_cfc_control_error[3] = {0.f, 0.f, 0.f};
float rl_cfc_control_input_error[3] = {0.f, 0.f, 0.f};
float rl_cfc_control_input_velocity[3] = {0.f, 0.f, 0.f};
float rl_cfc_control_feedback_rpm[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_motor_state[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_z_sign = RL_CFC_OBS_Z_SIGN;
float rl_cfc_control_vz_sign = RL_CFC_OBS_VZ_SIGN;
float rl_cfc_control_mean_policy_rpm = 0.f;
float rl_cfc_control_raw_action[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_action[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_policy_raw_action[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_obs[NUM_STATES] = {0.f};
float rl_cfc_control_next_gate[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_gate_yaw[8] = {
  0.5f * (float)M_PI,
  (float)M_PI,
  0.5f * (float)M_PI,
  0.0f,
  -0.5f * (float)M_PI,
  -(float)M_PI,
  -0.5f * (float)M_PI,
  0.0f
};
float rl_cfc_control_dist_to_target = 0.f;
static uint32_t rl_cfc_control_last_periodic_start_us = 0U;

static bool rl_cfc_control_has_autonomous_authority(void)
{
  const uint8_t mode = autopilot_get_mode();
  return mode == AP_MODE_NAV || mode == AP_MODE_GUIDED;
}
static bool rl_cfc_control_has_previous_gate_projection = false;
static float rl_cfc_control_previous_gate_projection = 0.f;
#if !RL_CFC_HAS_BEBOP_ACTUATORS
static float rl_cfc_control_sim_rpm_est[4] = {
  RL_CFC_TRAIN_HOVER_RPM,
  RL_CFC_TRAIN_HOVER_RPM,
  RL_CFC_TRAIN_HOVER_RPM,
  RL_CFC_TRAIN_HOVER_RPM
};
#endif

/*
 * Figure-eight gate centers from the RL environment.
 * RL world uses z-down. This table is stored in Paparazzi ENU as:
 *   enu = {rl_y, rl_x, -rl_z}
 * so enu_to_ned_vec(current - gate) recreates the raw RL world position error.
 */
static const rl_vec3_t rl_figure_eight_waypoints[] = {
  {-1.5f,  1.5f, RL_CFC_TARGET_ALT_M},
  { 0.0f,  0.0f, RL_CFC_TARGET_ALT_M},
  { 1.5f, -1.5f, RL_CFC_TARGET_ALT_M},
  { 3.0f,  0.0f, RL_CFC_TARGET_ALT_M},
  { 1.5f,  1.5f, RL_CFC_TARGET_ALT_M},
  { 0.0f,  0.0f, RL_CFC_TARGET_ALT_M},
  {-1.5f, -1.5f, RL_CFC_TARGET_ALT_M},
  {-3.0f,  0.0f, RL_CFC_TARGET_ALT_M},
};
static const unsigned int rl_num_figure_eight_waypoints =
    sizeof(rl_figure_eight_waypoints) / sizeof(rl_figure_eight_waypoints[0]);

/* Network outputs use Paparazzi's QUAD_X motor order: front-left, front-right, back-right, back-left. */
static const uint8_t rl_cfc_net_to_phys_motor[4] = {
  MOTOR_FRONT_LEFT,
  MOTOR_FRONT_RIGHT,
  MOTOR_BACK_RIGHT,
  MOTOR_BACK_LEFT,
};

static rl_vec3_t get_figure_eight_waypoint(unsigned int index)
{
#if defined(WP_RL_F8_1) && defined(WP_RL_F8_2) && defined(WP_RL_F8_3) && defined(WP_RL_F8_4) && \
    defined(WP_RL_F8_5) && defined(WP_RL_F8_6) && defined(WP_RL_F8_7) && defined(WP_RL_F8_8)
  static const uint8_t wp_ids[] = {
    WP_RL_F8_1, WP_RL_F8_2, WP_RL_F8_3, WP_RL_F8_4,
    WP_RL_F8_5, WP_RL_F8_6, WP_RL_F8_7, WP_RL_F8_8
  };
  const uint8_t wp_id = wp_ids[index % (sizeof(wp_ids) / sizeof(wp_ids[0]))];
  return (rl_vec3_t){waypoint_get_x(wp_id), waypoint_get_y(wp_id), waypoint_get_alt(wp_id)};
#else
  return rl_figure_eight_waypoints[index % rl_num_figure_eight_waypoints];
#endif
}

__attribute__((weak)) rl_vec3_t rl_cfc_get_position_m(void)
{
  const struct EnuCoor_f *pos = stateGetPositionEnu_f();
  return (rl_vec3_t){pos->x, pos->y, pos->z};
}

__attribute__((weak)) rl_vec3_t rl_cfc_get_velocity_mps(void)
{
  const struct EnuCoor_f *vel = stateGetSpeedEnu_f();
  return (rl_vec3_t){vel->x, vel->y, vel->z};
}

__attribute__((weak)) rl_euler_t rl_cfc_get_attitude_rad(void)
{
  const struct FloatEulers *att = stateGetNedToBodyEulers_f();
  return (rl_euler_t){att->phi, att->theta, att->psi};
}

__attribute__((weak)) rl_vec3_t rl_cfc_get_body_rates_radps(void)
{
  const struct FloatRates *rates = stateGetBodyRates_f();
  return (rl_vec3_t){rates->p, rates->q, rates->r};
}

__attribute__((weak)) void rl_cfc_get_motor_feedback_rpm(float omega_rpm[4])
{
#if RL_CFC_HAS_BEBOP_ACTUATORS
  for (uint8_t i = 0U; i < 4U; i++) {
    omega_rpm[i] = (float)actuators_bebop.rpm_obs[rl_cfc_net_to_phys_motor[i]];
  }
#else
  omega_rpm[0] = rl_cfc_control_sim_rpm_est[0];
  omega_rpm[1] = rl_cfc_control_sim_rpm_est[1];
  omega_rpm[2] = rl_cfc_control_sim_rpm_est[2];
  omega_rpm[3] = rl_cfc_control_sim_rpm_est[3];
#endif
}

static float clampf(float x, float lo, float hi)
{
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

#if !RL_CFC_HAS_BEBOP_ACTUATORS
static void reset_sim_motor_feedback_rpm(float rpm)
{
  rpm = clampf(rpm, RL_CFC_MIN_RPM, RL_CFC_MAX_RPM);
  for (uint8_t i = 0U; i < 4U; i++) {
    rl_cfc_control_sim_rpm_est[i] = rpm;
  }
}

static void update_sim_motor_feedback_rpm(void)
{
  const float dt = CFC_TIMESPAN;
  const float tau = RL_CFC_SIM_MOTOR_TAU_S > 0.f ? RL_CFC_SIM_MOTOR_TAU_S : dt;
  const float alpha = clampf(1.f - expf(-dt / tau), 0.f, 1.f);

  for (uint8_t i = 0U; i < 4U; i++) {
    const float rpm_target = clampf(rl_cfc_control_last_rpm[i], RL_CFC_MIN_RPM, RL_CFC_MAX_RPM);
    rl_cfc_control_sim_rpm_est[i] += alpha * (rpm_target - rl_cfc_control_sim_rpm_est[i]);
  }
}
#endif

static float wrap_pi(float x)
{
  while (x > (float)M_PI) {
    x -= 2.0f * (float)M_PI;
  }
  while (x < -(float)M_PI) {
    x += 2.0f * (float)M_PI;
  }
  return x;
}

static float rpm_to_motor_state(float rpm)
{
  return 2.0f * (clampf(rpm, RL_CFC_MIN_RPM, RL_CFC_MAX_RPM) - RL_CFC_MIN_RPM) /
         (RL_CFC_MAX_RPM - RL_CFC_MIN_RPM) - 1.0f;
}

static rl_vec3_t gate_frame_vec(const rl_vec3_t vec, float gate_yaw)
{
  const float c = cosf(gate_yaw);
  const float s = sinf(gate_yaw);
  return (rl_vec3_t){vec.x * c + vec.y * s, -vec.x * s + vec.y * c, vec.z};
}

static rl_vec3_t enu_to_ned_vec(const rl_vec3_t enu)
{
  return (rl_vec3_t){enu.y, enu.x, -enu.z};
}

static int32_t rpm_to_command(float rpm)
{
  rpm = clampf(rpm, RL_CFC_MIN_RPM, RL_CFC_MAX_RPM);
  return (int32_t)(rpm >= 0.f ? rpm + 0.5f : rpm - 0.5f);
}

static void update_motor_command_debug(void)
{
  int32_t sum = 0;
  for (uint8_t i = 0U; i < 4U; i++) {
    sum += rpm_to_command(rl_cfc_control_last_rpm[i]);
  }
  rl_cfc_control_raw_mean_rpm = sum / 4;
}

#if !RL_CFC_HAS_BEBOP_ACTUATORS
static int32_t rpm_to_nps_pprz_command(float rpm)
{
  /*
   * While the neural aerodynamic model is active, the NPS actuator channel is
   * a direct normalized-RPM transport: 0 -> 0 rpm, 1 -> max training RPM.
   * The stock Gazebo thrust interpretation is used only before RL activation.
   */
  const float normalized_rpm = clampf(rpm, 0.f, RL_CFC_MAX_RPM) / RL_CFC_MAX_RPM;
  const float pprz = normalized_rpm * (float)MAX_PPRZ;
  return (int32_t)(pprz + 0.5f);
}
#endif

static void apply_motor_rpm(uint8_t motor_idx, int32_t rpm)
{
#if !RL_CFC_HAS_BEBOP_ACTUATORS
  if (motor_idx < MOTOR_MIXING_NB_MOTOR) {
    motor_mixing.commands[motor_idx] = rpm_to_nps_pprz_command((float)rpm);
  }
#else
  actuators_bebop_set(motor_idx, (int16_t)rpm);
#endif
}

static float waypoint_dist2(const rl_vec3_t pos, const rl_vec3_t wp)
{
  const float dx = wp.x - pos.x;
  const float dy = wp.y - pos.y;
  const float dz = wp.z - pos.z;
  return dx * dx + dy * dy + dz * dz;
}

#if RL_CFC_START_WAYPOINT_INDEX < 0
static float waypoint_xy_dist2(const rl_vec3_t a, const rl_vec3_t b)
{
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  return dx * dx + dy * dy;
}
#endif

static unsigned int choose_initial_waypoint(const rl_vec3_t pos)
{
#if RL_CFC_START_WAYPOINT_INDEX >= 0
  (void)pos;
  return (unsigned int)RL_CFC_START_WAYPOINT_INDEX % rl_num_figure_eight_waypoints;
#else
  unsigned int best_any = 0U;
  unsigned int best_behind = 0U;
  float best_any_d2 = 1.0e9f;
  float best_behind_d2 = 1.0e9f;
  bool found_behind = false;
  const rl_vec3_t rl_pos = enu_to_ned_vec(pos);

  for (unsigned int i = 0U; i < rl_num_figure_eight_waypoints; i++) {
    const rl_vec3_t wp = get_figure_eight_waypoint(i);
    const float d2 = waypoint_xy_dist2(pos, wp);
    if (d2 < best_any_d2) {
      best_any_d2 = d2;
      best_any = i;
    }

    const rl_vec3_t rl_wp = enu_to_ned_vec(wp);
    const float gate_yaw = rl_cfc_control_gate_yaw[i];
    const float projected =
        (rl_pos.x - rl_wp.x) * cosf(gate_yaw)
        + (rl_pos.y - rl_wp.y) * sinf(gate_yaw);
    if (projected < 0.f && d2 < best_behind_d2) {
      best_behind_d2 = d2;
      best_behind = i;
      found_behind = true;
    }
  }

  return found_behind ? best_behind : best_any;
#endif
}

static void update_target_debug(const rl_vec3_t pos, const rl_vec3_t wp)
{
  rl_cfc_control_target[0] = wp.x;
  rl_cfc_control_target[1] = wp.y;
  rl_cfc_control_target[2] = wp.z;

  rl_cfc_control_error[0] = wp.x - pos.x;
  rl_cfc_control_error[1] = wp.y - pos.y;
  rl_cfc_control_error[2] = wp.z - pos.z;
  rl_cfc_control_dist_to_target = sqrtf(waypoint_dist2(pos, wp));
}
static void update_next_gate_debug(unsigned int waypoint_index)
{
  const rl_vec3_t current_wp = get_figure_eight_waypoint(waypoint_index);
  const rl_vec3_t next_wp = get_figure_eight_waypoint(waypoint_index + 1U);
  rl_vec3_t next_rel = {
    next_wp.x - current_wp.x,
    next_wp.y - current_wp.y,
    next_wp.z - current_wp.z
  };
  const float gate_yaw = rl_cfc_control_gate_yaw[waypoint_index % rl_num_figure_eight_waypoints];
  if (rl_cfc_control_use_ned_input) {
    next_rel = enu_to_ned_vec(next_rel);
  }
  next_rel = gate_frame_vec(next_rel, gate_yaw);

  rl_cfc_control_next_gate[0] = next_rel.x;
  rl_cfc_control_next_gate[1] = next_rel.y;
  rl_cfc_control_next_gate[2] = next_rel.z;
  rl_cfc_control_next_gate[3] =
      wrap_pi(rl_cfc_control_gate_yaw[(waypoint_index + 1U) % rl_num_figure_eight_waypoints] - gate_yaw);
}

static float gate_plane_projection(const rl_vec3_t pos, const rl_vec3_t wp, float gate_yaw)
{
  const rl_vec3_t rl_pos = enu_to_ned_vec(pos);
  const rl_vec3_t rl_wp = enu_to_ned_vec(wp);
  return (rl_pos.x - rl_wp.x) * cosf(gate_yaw) +
         (rl_pos.y - rl_wp.y) * sinf(gate_yaw);
}

static bool inside_gate_window(const rl_vec3_t pos, const rl_vec3_t wp)
{
  const rl_vec3_t rl_pos = enu_to_ned_vec(pos);
  const rl_vec3_t rl_wp = enu_to_ned_vec(wp);
  const float half_gate = 0.5f * RL_CFC_GATE_SIZE_M;
  return fabsf(rl_pos.x - rl_wp.x) < half_gate &&
         fabsf(rl_pos.y - rl_wp.y) < half_gate &&
         fabsf(rl_pos.z - rl_wp.z) < half_gate;
}

static void reset_target_debug(void)
{
  for (unsigned int i = 0; i < 3U; i++) {
    rl_cfc_control_target[i] = 0.f;
    rl_cfc_control_error[i] = 0.f;
    rl_cfc_control_input_error[i] = 0.f;
    rl_cfc_control_input_velocity[i] = 0.f;
  }
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_motor_state[i] = 0.f;
    rl_cfc_control_raw_action[i] = 0.f;
    rl_cfc_control_action[i] = 0.f;
    rl_cfc_control_policy_raw_action[i] = 0.f;
    rl_cfc_control_next_gate[i] = 0.f;
  }
  for (unsigned int i = 0; i < NUM_STATES; i++) {
    rl_cfc_control_obs[i] = 0.f;
  }
  rl_cfc_control_dist_to_target = 0.f;
}

static void maybe_advance_waypoint(const rl_vec3_t pos)
{
  const rl_vec3_t wp = get_figure_eight_waypoint(rl_cfc_control_waypoint_index);
  const float gate_yaw = rl_cfc_control_gate_yaw[rl_cfc_control_waypoint_index % rl_num_figure_eight_waypoints];
  const float gate_projection = gate_plane_projection(pos, wp, gate_yaw);
  const bool passed_gate_plane =
      rl_cfc_control_has_previous_gate_projection &&
      rl_cfc_control_previous_gate_projection < 0.f &&
      gate_projection > 0.f;
  const bool gate_passed = passed_gate_plane && inside_gate_window(pos, wp);
  if (gate_passed) {
    rl_cfc_control_waypoint_index = (rl_cfc_control_waypoint_index + 1U) % rl_num_figure_eight_waypoints;
    // rl_cfc_reset();
    rl_cfc_control_has_previous_gate_projection = false;
  }
  const rl_vec3_t active_wp = get_figure_eight_waypoint(rl_cfc_control_waypoint_index);
  const float active_gate_yaw =
      rl_cfc_control_gate_yaw[rl_cfc_control_waypoint_index % rl_num_figure_eight_waypoints];
  rl_cfc_control_previous_gate_projection = gate_plane_projection(pos, active_wp, active_gate_yaw);
  rl_cfc_control_has_previous_gate_projection = true;
  update_target_debug(pos, active_wp);
  update_next_gate_debug(rl_cfc_control_waypoint_index);
}

void rl_cfc_control_init(void)
{
  rl_cfc_control_enabled = false;
  rl_cfc_control_waypoint_index = 0U;
  rl_cfc_control_periodic_count = 0U;
  rl_cfc_control_periodic_dt_us = 0U;
  rl_cfc_control_sensor_read_time_us = 0U;
  rl_cfc_control_inference_time_us = 0U;
  rl_cfc_control_total_time_us = 0U;
  rl_cfc_control_last_periodic_start_us = 0U;
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_last_norm[i] = 0.f;
    rl_cfc_control_last_rpm[i] = 0.f;
    rl_cfc_control_motor_rpm_cmd[i] = 0;
    rl_cfc_control_applied_rpm_cmd[i] = 0;
    rl_cfc_control_feedback_rpm[i] = 0.f;
    rl_cfc_control_motor_state[i] = 0.f;
    rl_cfc_control_raw_action[i] = 0.f;
    rl_cfc_control_action[i] = 0.f;
    rl_cfc_control_policy_raw_action[i] = 0.f;
  }
#if !RL_CFC_HAS_BEBOP_ACTUATORS
  reset_sim_motor_feedback_rpm(RL_CFC_TRAIN_HOVER_RPM);
#endif
  rl_cfc_control_raw_mean_rpm = 0;
  rl_cfc_control_mean_policy_rpm = 0.f;
  rl_cfc_control_has_previous_gate_projection = false;
  rl_cfc_control_previous_gate_projection = 0.f;
  reset_target_debug();
  rl_cfc_reset();
#ifdef MODULE_LOGGER_FILE_ID
  logger_file_start();
#endif
}

void rl_cfc_control_start(void)
{
  /*
   * Neural motor authority is restricted to autonomous NAV/GUIDED modes. A
   * safety pilot selecting ATT (or any failsafe/manual mode) must always regain
   * the normal stabilization/mixer path without waiting for the flight plan.
   */
  if (!rl_cfc_control_has_autonomous_authority()) {
    rl_cfc_control_stop();
    return;
  }

  rl_cfc_control_enabled = true;
  rl_cfc_control_periodic_count = 0U;
  rl_cfc_control_periodic_dt_us = 0U;
  rl_cfc_control_sensor_read_time_us = 0U;
  rl_cfc_control_inference_time_us = 0U;
  rl_cfc_control_total_time_us = 0U;
  rl_cfc_control_last_periodic_start_us = 0U;
  const float hover_rpm = clampf(RL_CFC_TRAIN_HOVER_RPM, RL_CFC_MIN_RPM, RL_CFC_MAX_RPM);
  const float hover_norm = (hover_rpm - RL_CFC_MIN_RPM) / (RL_CFC_MAX_RPM - RL_CFC_MIN_RPM);
  const float hover_action = hover_norm;
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_last_norm[i] = hover_norm;
    rl_cfc_control_last_rpm[i] = hover_rpm;
    rl_cfc_control_motor_rpm_cmd[i] = rpm_to_command(hover_rpm);
    rl_cfc_control_applied_rpm_cmd[i] = rpm_to_command(hover_rpm);
    rl_cfc_control_feedback_rpm[i] = hover_rpm;
    rl_cfc_control_motor_state[i] = rpm_to_motor_state(hover_rpm);
    rl_cfc_control_policy_raw_action[i] = hover_action;
    rl_cfc_control_raw_action[i] = hover_action;
    rl_cfc_control_action[i] = hover_action;
  }
#if !RL_CFC_HAS_BEBOP_ACTUATORS
  reset_sim_motor_feedback_rpm(hover_rpm);
#endif
  rl_cfc_control_raw_mean_rpm = rpm_to_command(hover_rpm);
  rl_cfc_control_mean_policy_rpm = hover_rpm;
  rl_cfc_control_has_previous_gate_projection = false;
  rl_cfc_control_previous_gate_projection = 0.f;
  const rl_vec3_t start_pos = rl_cfc_get_position_m();
  rl_cfc_control_waypoint_index = choose_initial_waypoint(start_pos);
  update_target_debug(start_pos, get_figure_eight_waypoint(rl_cfc_control_waypoint_index));
  update_next_gate_debug(rl_cfc_control_waypoint_index);
  rl_cfc_reset();
#ifdef MODULE_LOGGER_FILE_ID
  logger_file_start();
#endif
}

void rl_cfc_control_stop(void)
{
  rl_cfc_control_enabled = false;
  rl_cfc_control_periodic_dt_us = 0U;
  rl_cfc_control_sensor_read_time_us = 0U;
  rl_cfc_control_inference_time_us = 0U;
  rl_cfc_control_total_time_us = 0U;
  rl_cfc_control_last_periodic_start_us = 0U;
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_last_norm[i] = 0.f;
    rl_cfc_control_last_rpm[i] = 0.f;
    rl_cfc_control_motor_rpm_cmd[i] = 0;
    rl_cfc_control_applied_rpm_cmd[i] = 0;
    rl_cfc_control_feedback_rpm[i] = 0.f;
    rl_cfc_control_motor_state[i] = 0.f;
    rl_cfc_control_policy_raw_action[i] = 0.f;
    rl_cfc_control_raw_action[i] = 0.f;
    rl_cfc_control_action[i] = 0.f;
  }
#if !RL_CFC_HAS_BEBOP_ACTUATORS
  reset_sim_motor_feedback_rpm(RL_CFC_TRAIN_HOVER_RPM);
#endif
  rl_cfc_control_raw_mean_rpm = 0;
  rl_cfc_control_mean_policy_rpm = 0.f;
  rl_cfc_control_has_previous_gate_projection = false;
  rl_cfc_control_previous_gate_projection = 0.f;
  reset_target_debug();
  rl_cfc_reset();
}

void rl_cfc_control_periodic(void)
{
  if (!rl_cfc_control_enabled) {
    return;
  }
  if (!rl_cfc_control_has_autonomous_authority()) {
    rl_cfc_control_stop();
    return;
  }

  const uint32_t periodic_start_us = get_sys_time_usec();
  if (rl_cfc_control_last_periodic_start_us != 0U) {
    rl_cfc_control_periodic_dt_us = periodic_start_us - rl_cfc_control_last_periodic_start_us;
  } else {
    rl_cfc_control_periodic_dt_us = 0U;
  }
  rl_cfc_control_last_periodic_start_us = periodic_start_us;
  rl_cfc_control_periodic_count++;

  const uint32_t sensor_read_start_us = get_sys_time_usec();
  const rl_vec3_t pos = rl_cfc_get_position_m();
  maybe_advance_waypoint(pos);

  const rl_vec3_t vel = rl_cfc_get_velocity_mps();
  const rl_vec3_t err = {
    rl_cfc_control_error[0],
    rl_cfc_control_error[1],
    rl_cfc_control_error[2]
  };
  rl_vec3_t pos_gate = {-err.x, -err.y, -err.z};
  rl_vec3_t vel_gate = vel;
  if (rl_cfc_control_use_ned_input) {
    pos_gate = enu_to_ned_vec(pos_gate);
    vel_gate = enu_to_ned_vec(vel_gate);
  }
  const float gate_yaw = rl_cfc_control_gate_yaw[rl_cfc_control_waypoint_index % rl_num_figure_eight_waypoints];
  pos_gate = gate_frame_vec(pos_gate, gate_yaw);
  vel_gate = gate_frame_vec(vel_gate, gate_yaw);
  pos_gate.z *= rl_cfc_control_z_sign >= 0.f ? 1.f : -1.f;
  vel_gate.z *= rl_cfc_control_vz_sign >= 0.f ? 1.f : -1.f;
  rl_cfc_control_input_error[0] = pos_gate.x;
  rl_cfc_control_input_error[1] = pos_gate.y;
  rl_cfc_control_input_error[2] = pos_gate.z;
  rl_cfc_control_input_velocity[0] = vel_gate.x;
  rl_cfc_control_input_velocity[1] = vel_gate.y;
  rl_cfc_control_input_velocity[2] = vel_gate.z;

  const rl_euler_t att = rl_cfc_get_attitude_rad();
  const rl_vec3_t rates = rl_cfc_get_body_rates_radps();
  float omega[4];
  rl_cfc_get_motor_feedback_rpm(omega);
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_feedback_rpm[i] = omega[i];
    rl_cfc_control_motor_state[i] = rpm_to_motor_state(omega[i]);
  }
  rl_cfc_control_sensor_read_time_us = get_sys_time_usec() - sensor_read_start_us;

  float obs[NUM_STATES] = {
    pos_gate.x, pos_gate.y, pos_gate.z,
    vel_gate.x, vel_gate.y, vel_gate.z,
    RL_CFC_OBS_ROLL_SIGN * att.phi,
    RL_CFC_OBS_PITCH_SIGN * att.theta,
    RL_CFC_OBS_YAW_SIGN * wrap_pi(att.psi - gate_yaw),
    RL_CFC_OBS_P_SIGN * rates.x,
    RL_CFC_OBS_Q_SIGN * rates.y,
    RL_CFC_OBS_R_SIGN * rates.z,
    rl_cfc_control_motor_state[0],
    rl_cfc_control_motor_state[1],
    rl_cfc_control_motor_state[2],
    rl_cfc_control_motor_state[3],
    rl_cfc_control_next_gate[0],
    rl_cfc_control_next_gate[1],
    rl_cfc_control_next_gate[2],
    rl_cfc_control_next_gate[3]
  };
  rl_cfc_add_nps_observation_noise(obs);
  for (unsigned int i = 0; i < NUM_STATES; i++) {
    rl_cfc_control_obs[i] = obs[i];
  }

  float action[NUM_CONTROLS];
  const uint32_t inference_start_us = get_sys_time_usec();
  rl_cfc_control(obs, action);
  rl_cfc_control_inference_time_us = get_sys_time_usec() - inference_start_us;

  float normalized_cmd[NUM_CONTROLS];
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_policy_raw_action[i] = action[i];
    rl_cfc_control_raw_action[i] = rl_cfc_last_raw_control[i];
    rl_cfc_control_action[i] = clampf(action[i], 0.f, 1.f);
    normalized_cmd[i] = rl_cfc_control_action[i];
  }
  float rpm_sum = 0.f;
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_last_norm[i] = clampf(normalized_cmd[i], 0.f, 1.f);
    rl_cfc_control_last_rpm[i] =
        RL_CFC_MIN_RPM +
        rl_cfc_control_last_norm[i] * (RL_CFC_MAX_RPM - RL_CFC_MIN_RPM);
    rl_cfc_control_motor_rpm_cmd[rl_cfc_net_to_phys_motor[i]] =
        rpm_to_command(rl_cfc_control_last_rpm[i]);
    rpm_sum += rl_cfc_control_last_rpm[i];
  }
#if !RL_CFC_HAS_BEBOP_ACTUATORS
  update_sim_motor_feedback_rpm();
#endif
  rl_cfc_control_mean_policy_rpm = rpm_sum * 0.25f;
  update_motor_command_debug();
  rl_cfc_control_total_time_us = get_sys_time_usec() - periodic_start_us;
}

void rl_cfc_control_apply_motor_rpm(bool motors_on)
{
  /*
   * command_laws run the standard mixer immediately before this function.
   * On an autonomous -> manual transition, stop RL and leave those mixer
   * commands intact instead of overwriting them with the last network RPMs.
   */
  if (rl_cfc_control_enabled && !rl_cfc_control_has_autonomous_authority()) {
    rl_cfc_control_stop();
  }

  if (!motors_on || !rl_cfc_control_enabled) {
    for (uint8_t i = 0U; i < 4U; i++) {
      rl_cfc_control_applied_rpm_cmd[i] = 0;
    }
    return;
  }

  update_motor_command_debug();
  for (uint8_t i = 0U; i < 4U; i++) {
    const uint8_t phys_idx = rl_cfc_net_to_phys_motor[i];
    const int32_t rpm = rpm_to_command(rl_cfc_control_last_rpm[i]);
    rl_cfc_control_applied_rpm_cmd[phys_idx] = rpm;
    apply_motor_rpm(phys_idx, rpm);
  }
}
