#ifndef NN_CFC_CONTROL_H
#define NN_CFC_CONTROL_H

#include <stdbool.h>
#include <stdint.h>

extern bool nn_cfc_control_enabled;
extern unsigned int nn_cfc_control_waypoint_index;
extern float nn_cfc_control_last_norm[4];
extern float nn_cfc_control_last_rpm[4];
extern int32_t nn_cfc_control_motor_rpm_cmd[4];
extern int32_t nn_cfc_control_applied_rpm_cmd[4];
extern int32_t nn_cfc_control_raw_mean_rpm;
extern unsigned int nn_cfc_control_periodic_count;
extern uint32_t nn_cfc_control_periodic_dt_us;
extern uint32_t nn_cfc_control_sensor_read_time_us;
extern uint32_t nn_cfc_control_inference_time_us;
extern uint32_t nn_cfc_control_total_time_us;
extern unsigned int nn_cfc_control_waypoint_switch_count;
extern float nn_cfc_control_reached_radius_m;
extern float nn_cfc_control_target[3];
extern float nn_cfc_control_position[3];
extern float nn_cfc_control_error[3];
extern float nn_cfc_control_input_error[3];
extern float nn_cfc_control_input_velocity[3];
extern float nn_cfc_control_external_moment_nm[3];
extern float nn_cfc_control_feedback_rpm[4];
extern float nn_cfc_control_attitude[3];
extern float nn_cfc_control_body_rates[3];
extern float nn_cfc_control_dist_to_target;
extern float nn_cfc_control_dist_xy_to_target;
extern float nn_cfc_control_abs_z_error;
extern float nn_cfc_control_mean_norm;
extern int32_t nn_cfc_control_mean_rpm_cmd;
/* Debug: network raw inputs and outputs */
extern float nn_cfc_control_net_output_norm[4];
extern float nn_cfc_control_net_output_raw[4];
extern float nn_cfc_control_net_input_state[19];
extern float nn_cfc_control_net_input_normalized[19];
extern float nn_cfc_control_log_values[38];

void nn_cfc_control_init(void);
void nn_cfc_control_start(void);
void nn_cfc_control_stop(void);
void nn_cfc_control_periodic(void);
void nn_cfc_control_apply_motor_rpm(bool motors_on);

#endif
