#ifndef RL_CFC_CONTROL_H
#define RL_CFC_CONTROL_H

#include <stdbool.h>
#include <stdint.h>

extern bool rl_cfc_control_enabled;
extern unsigned int rl_cfc_control_waypoint_index;
extern float rl_cfc_control_last_norm[4];
extern float rl_cfc_control_last_rpm[4];
extern int32_t rl_cfc_control_motor_rpm_cmd[4];
extern int32_t rl_cfc_control_applied_rpm_cmd[4];
extern int32_t rl_cfc_control_raw_mean_rpm;
extern unsigned int rl_cfc_control_periodic_count;
extern uint32_t rl_cfc_control_periodic_dt_us;
extern uint32_t rl_cfc_control_sensor_read_time_us;
extern uint32_t rl_cfc_control_inference_time_us;
extern uint32_t rl_cfc_control_total_time_us;
extern bool rl_cfc_control_use_ned_input;
extern float rl_cfc_control_target[3];
extern float rl_cfc_control_error[3];
extern float rl_cfc_control_input_error[3];
extern float rl_cfc_control_input_velocity[3];
extern float rl_cfc_control_feedback_rpm[4];
extern float rl_cfc_control_motor_state[4];
extern float rl_cfc_control_z_sign;
extern float rl_cfc_control_vz_sign;
extern float rl_cfc_control_mean_policy_rpm;
extern float rl_cfc_control_raw_action[4];
extern float rl_cfc_control_action[4];
extern float rl_cfc_control_policy_raw_action[4];
extern float rl_cfc_control_obs[20];
extern float rl_cfc_control_next_gate[4];
extern float rl_cfc_control_gate_yaw[8];
extern float rl_cfc_control_dist_to_target;

void rl_cfc_control_init(void);
void rl_cfc_control_start(void);
void rl_cfc_control_stop(void);
void rl_cfc_control_periodic(void);
void rl_cfc_control_apply_motor_rpm(bool motors_on);

#endif
