
import numpy as np
# from quadrotor_env_class_robin import Quadcopter3DGates, Quadcopter3DGatesGymEnv
import numpy as np
import gym
import random
import torch
import stable_baselines3
from stable_baselines3.common.vec_env import VecEnv
import numpy as np
from sympy import *
import gym
import sys
from gymnasium import spaces

# from analyze_flight_data/SystemIdentification.ipynb
# 5inch drone:
params_5inch = {
    'k_w': 2.49e-06, 'k_x': 4.85e-05, 'k_y': 7.28e-05,
    'k_p1': 6.55e-05, 'k_p2': 6.61e-05, 'k_p3': 6.36e-05, 'k_p4': 6.67e-05,
    'k_q1': 5.28e-05, 'k_q2': 5.86e-05, 'k_q3': 5.05e-05, 'k_q4': 5.89e-05,
    'k_r1': 1.07e-02, 'k_r2': 1.07e-02, 'k_r3': 1.07e-02, 'k_r4': 1.07e-02, 'k_r5': 1.97e-03, 'k_r6': 1.97e-03,
    'k_r7': 1.97e-03, 'k_r8': 1.97e-03,
    'w_min': 238.49, 'w_max': 3295.50, 'k': 0.95, 'tau': 0.04
}
params_normalized_5inch = {
    'k_wn': 2.71e+01, 'k_xn': 1.60e-01, 'k_yn': 2.40e-01,
    'k_p1n': 7.11e+02, 'k_p2n': 7.18e+02, 'k_p3n': 6.91e+02, 'k_p4n': 7.24e+02,
    'k_q1n': 5.73e+02, 'k_q2n': 6.37e+02, 'k_q3n': 5.48e+02, 'k_q4n': 6.40e+02,
    'k_rn': 3.52e+01, 'k_rdn': 6.49e+00,
    'w_min': 238.49, 'w_max': 3295.50, 'k': 0.95, 'tau': 0.04
}
randomization_fixed_params_5inch = lambda num: {key: np.repeat(value, num) for key, value in params_5inch.items()}
# 10% randomization - uniform distribution between 0.9 and 1.1 of the original value. However, for k we must ensure that it is between 0 and 1
randomization_5inch_10_percent = lambda num: {
    key: np.random.uniform(value * 0.9, value * 1.1, num) if key != 'k' else np.random.uniform(value * 0.9, min(value * 1.1, 1), num)
    for key, value in params_5inch.items()}
# 20% randomization
randomization_5inch_20_percent = lambda num: {
    key: np.random.uniform(value * 0.8, value * 1.2, num) if key != 'k' else np.random.uniform(value * 0.8, min(value * 1.2, 1), num)
    for key, value in params_5inch.items()}
# 30% randomization
randomization_5inch_30_percent = lambda num: {
    key: np.random.uniform(value * 0.7, value * 1.3, num) if key != 'k' else np.random.uniform(value * 0.7, min(value * 1.3, 1), num)
    for key, value in params_5inch.items()}

SEED = 1

# Using default parameters from original code
RANDOMIZATION = randomization_5inch_30_percent  #### not default
state_history = 0
action_history = 0
history_step_size = 1
param_input = False
param_input_noise = 0.0

print("python version:", sys.version)
print("stable_baselines3 version:", stable_baselines3.__version__)
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("cudnn version:", torch.backends.cudnn.version())

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


print("device:", device)


# set torch default device
torch.set_default_device(device)

# Equations of motion 3D quadcopter from https://arxiv.org/pdf/2304.13460.pdf
state = symbols('x y z v_x v_y v_z phi theta psi p q r w1 w2 w3 w4')
x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, w1, w2, w3, w4 = state
control = symbols('U_1 U_2 U_3 U_4')  # normalized motor commands between [-1,1]
u1, u2, u3, u4 = control

g = 9.81
params = symbols(
    'k_x, k_y, k_w, k_p1, k_p2, k_p3, k_p4, k_q1, k_q2, k_q3, k_q4, k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8, tau, k, w_min, w_max')
k_x, k_y, k_w, k_p1, k_p2, k_p3, k_p4, k_q1, k_q2, k_q3, k_q4, k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8, tau, k, w_min, w_max = params

# Rotation matrix
Rx = Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
Ry = Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
Rz = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
R = Rz * Ry * Rx

# Body velocity
vbx, vby, vbz = R.T @ Matrix([vx, vy, vz])

# normalized motor speeds to rad/s
w_min_n = 0.
w_max_n = 3000.
W1 = (w1 + 1) / 2 * (w_max_n - w_min_n) + w_min_n
W2 = (w2 + 1) / 2 * (w_max_n - w_min_n) + w_min_n
W3 = (w3 + 1) / 2 * (w_max_n - w_min_n) + w_min_n
W4 = (w4 + 1) / 2 * (w_max_n - w_min_n) + w_min_n

# motor commands scaled to [0,1]
U1 = (u1 + 1) / 2
U2 = (u2 + 1) / 2
U3 = (u3 + 1) / 2
U4 = (u4 + 1) / 2

# first order delay:
# the steadystate rpm motor response to the motor command U is described by:
# Wc = (w_max-w_min)*sqrt(k U**2 + (1-k)*U) + w_min
Wc1 = (w_max - w_min) * sqrt(k * U1 ** 2 + (1 - k) * U1) + w_min
Wc2 = (w_max - w_min) * sqrt(k * U2 ** 2 + (1 - k) * U2) + w_min
Wc3 = (w_max - w_min) * sqrt(k * U3 ** 2 + (1 - k) * U3) + w_min
Wc4 = (w_max - w_min) * sqrt(k * U4 ** 2 + (1 - k) * U4) + w_min

# rad/s
d_W1 = (Wc1 - W1) / tau
d_W2 = (Wc2 - W2) / tau
d_W3 = (Wc3 - W3) / tau
d_W4 = (Wc4 - W4) / tau

# normalized motor speeds d/dt[W - w_min_n)/(w_max_n-w_min_n)*2 - 1]
d_w1 = d_W1 / (w_max_n - w_min_n) * 2
d_w2 = d_W2 / (w_max_n - w_min_n) * 2
d_w3 = d_W3 / (w_max_n - w_min_n) * 2
d_w4 = d_W4 / (w_max_n - w_min_n) * 2

# Thrust and Drag
T = -k_w * (W1 ** 2 + W2 ** 2 + W3 ** 2 + W4 ** 2)
Dx = -k_x * vbx * (W1 + W2 + W3 + W4)
Dy = -k_y * vby * (W1 + W2 + W3 + W4)

# Moments
Mx = -k_p1 * W1 ** 2 - k_p2 * W2 ** 2 + k_p3 * W3 ** 2 + k_p4 * W4 ** 2
My = -k_q1 * W1 ** 2 + k_q2 * W2 ** 2 - k_q3 * W3 ** 2 + k_q4 * W4 ** 2
Mz = -k_r1 * W1 + k_r2 * W2 + k_r3 * W3 - k_r4 * W4 - k_r5 * d_W1 + k_r6 * d_W1 + k_r7 * d_W1 - k_r8 * d_W1

# Dynamics
d_x = vx
d_y = vy
d_z = vz

d_vx, d_vy, d_vz = Matrix([0, 0, g]) + R @ Matrix([Dx, Dy, T])

d_phi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
d_theta = q * cos(phi) - r * sin(phi)
d_psi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)

d_p = Mx
d_q = My
d_r = Mz

# State space model
f = [d_x, d_y, d_z, d_vx, d_vy, d_vz, d_phi, d_theta, d_psi, d_p, d_q, d_r, d_w1, d_w2, d_w3, d_w4]

# lambdify
f_func = lambdify((Array(state), Array(control), Array(params)), Array(f), 'numpy')

# PARAMETER ENCODING (used for parameter input)
# normalize thrust and moment constants by scaling with w_max
k_wn = k_w * (w_max ** 2)
k_pn = (k_p1 + k_p2 + k_p3 + k_p4) / 4 * (w_max ** 2)
k_qn = (k_q1 + k_q2 + k_q3 + k_q4) / 4 * (w_max ** 2)
k_rn = (k_r1 + k_r2 + k_r3 + k_r4) / 4 * (w_max)
k_rdn = (k_r5 + k_r6 + k_r7 + k_r8) / 4 * (w_max)

# normalize to [-1,1] based on min and max expected values
normalize = lambda x, x_min, x_max: 2 * (x - x_min) / (x_max - x_min) - 1
k_w_encoding = normalize(k_wn, 1.0e+01, 3.0e+01)
k_p_encoding = normalize(k_pn, 2.0e+02, 8.0e+02)
k_q_encoding = normalize(k_qn, 2.0e+02, 8.0e+02)
k_r_encoding = normalize(k_rn, 2.0e+01, 8.0e+01)
k_rd_encoding = normalize(k_rdn, 2.0e+00, 8.0e+00)
k_encoding = normalize(k, 0., 1.)
tau_encoding = normalize(tau, 0.01, 0.1)
w_min_encoding = normalize(w_min, 0, 500)
w_max_encoding = normalize(w_max, 3000, 5000)

# lambdify
param_encoding = lambdify((Array(params),), Array(
    [k_w_encoding, k_p_encoding, k_q_encoding, k_r_encoding, k_rd_encoding, k_encoding, tau_encoding, w_min_encoding,
     w_max_encoding]), 'numpy')

# Represent the hover target with a single gate-like frame so the observation
# layout stays compatible with the original environment.
gate_pos = np.array([[0.0, 0.0, -1.5]], dtype=np.float32)
gate_yaw = np.array([0.0], dtype=np.float32)
start_pos = np.array([1.5, 0.0, -1.5], dtype=np.float32)


class Quadcopter3DHover(VecEnv):
    def __init__(self,
                 num_envs,
                 randomization,
                 gates_pos=gate_pos,
                 gate_yaw=gate_yaw,
                 start_pos=start_pos,
                 gates_ahead=1,
                 pause_if_collision=False,
                 motor_limit=1.0,
                 initialize_at_random_gates=True,
                 initialize_uniform=False,
                 num_state_history=0,
                 num_action_history=0,
                 history_step_size=1,
                 param_input=False,
                 param_input_noise=0.,
                 seed=None,
                 low_obs=False,
                 start_xy_radius=2.5,
                 start_z_offset=1.0,
                 start_pos_jitter_xy=1.0,
                 start_pos_jitter_z=0.75,
                 start_velocity_limit=0.25,
                 start_tilt_limit=np.pi / 12,
                 start_yaw_limit=np.pi,
                 start_rate_limit=0.2,
                 start_motor_state_limit=0.75,
                 ):
        # set seed

        #
        # self.seed = seed
        # if self.seed is not None:
        #     np.random.seed(self.seed)
        #     torch.manual_seed(self.seed)
        self.seed(seed)
        self.low_obs = low_obs

        # Define the hover target
        self.start_pos = start_pos.astype(np.float32)
        self.gate_pos = gates_pos.astype(np.float32)
        self.gate_yaw = gate_yaw.astype(np.float32)
        self.num_gates = gates_pos.shape[0]
        self.gates_ahead = gates_ahead

        # Pause if collision
        self.pause_if_collision = pause_if_collision

        # print([p.name for p in params])
        # print( ['k_x', 'k_y', 'k_w', 'k_p1', 'k_p2', 'k_p3', 'k_p4', 'k_q1', 'k_q2', 'k_q3', 'k_q4', 'k_r1', 'k_r2', 'k_r3', 'k_r4', 'k_r5', 'k_r6', 'k_r7', 'k_r8', 'tau', 'k', 'w_min', 'w_max'])
        # raise ValueError('stop')

        # Domain randomization
        def rand_f(n):
            param_dict = randomization(n)
            return np.array([param_dict[p.name] for p in params]).T

        self.randomization = rand_f
        self.params = self.randomization(num_envs)

        # Motor limit
        self.motor_limit = motor_limit

        # Initialize at random gates
        self.initialize_at_random_gates = initialize_at_random_gates
        self.initialize_uniform = initialize_uniform
        self.start_xy_radius = start_xy_radius
        self.start_z_offset = start_z_offset
        self.start_pos_jitter_xy = start_pos_jitter_xy
        self.start_pos_jitter_z = start_pos_jitter_z
        self.start_velocity_limit = start_velocity_limit
        self.start_tilt_limit = start_tilt_limit
        self.start_yaw_limit = start_yaw_limit
        self.start_rate_limit = start_rate_limit
        self.start_motor_state_limit = start_motor_state_limit

        # state, action history
        self.num_state_history = num_state_history
        self.num_action_history = num_action_history
        self.history_step_size = history_step_size

        # param input
        self.param_input = param_input
        self.param_input_noise = param_input_noise
        # compute encoding with noise on the params
        if self.param_input:
            params_with_noise = self.params * np.random.uniform(1 - self.param_input_noise, 1 + self.param_input_noise,
                                                                size=self.params.shape)
            self.param_encoding = param_encoding(params_with_noise.T).T

        # Calculate relative gates
        # pos,yaw of gate i in reference frame of gate i-1 (assumes a looped track)
        self.gate_pos_rel = np.zeros((self.num_gates, 3), dtype=np.float32)
        self.gate_yaw_rel = np.zeros(self.num_gates, dtype=np.float32)
        for i in range(0, self.num_gates):
            self.gate_pos_rel[i] = self.gate_pos[i] - self.gate_pos[i - 1]
            # Rotation matrix
            R = np.array([
                [np.cos(self.gate_yaw[i - 1]), np.sin(self.gate_yaw[i - 1])],
                [-np.sin(self.gate_yaw[i - 1]), np.cos(self.gate_yaw[i - 1])]
            ])
            self.gate_pos_rel[i, 0:2] = R @ self.gate_pos_rel[i, 0:2]
            self.gate_yaw_rel[i] = self.gate_yaw[i] - self.gate_yaw[i - 1]
            # wrap yaw
            self.gate_yaw_rel[i] %= 2 * np.pi
            if self.gate_yaw_rel[i] > np.pi:
                self.gate_yaw_rel[i] -= 2 * np.pi
            elif self.gate_yaw_rel[i] < -np.pi:
                self.gate_yaw_rel[i] += 2 * np.pi

        # Define the target gate for each environment
        self.target_gates = np.zeros(num_envs, dtype=int)

        # action space: [cmd1, cmd2, cmd3, cmd4]
        # U = (u+1)/2 --> u = 2U-1
        u_lim = 2 * self.motor_limit - 1
        action_space = spaces.Box(low=-1, high=u_lim, shape=(4,), dtype=np.float32)

        # observation space: pos[G], vel[G], att[eulerB->G], rates[B], rpms, future_gates[G], future_gate_dirs[G]
        # [G] = reference frame aligned with target gate
        # [B] = body frame
        self.state_len = 16 + 4 * self.gates_ahead + 4 * self.num_action_history + 9 * self.param_input
        self.low_obs_state_len = 10 + 4 * self.gates_ahead + 4 * self.num_action_history + 9 * self.param_input
        if self.low_obs:
            self.obs_len = self.low_obs_state_len * (1 + self.num_state_history)
        else:
            self.obs_len = self.state_len * (1 + self.num_state_history)
        observation_space = spaces.Box(
                low=np.array([-np.inf] * self.obs_len),
                high=np.array([np.inf] * self.obs_len),
                shape=(self.obs_len,),
                dtype=np.float32
            )



        if seed is not None:
            action_space.seed(seed)
            observation_space.seed(seed)

        # Initialize the VecEnv
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        # world state: pos[W], vel[W], att[eulerB->W], rates[B], rpms
        self.world_states = np.zeros((num_envs, 16), dtype=np.float32)
        # observation state
        self.states = np.zeros((num_envs, self.obs_len), dtype=np.float32)
        # state history tracking
        num_hist = 40
        self.state_hist = np.zeros((num_envs, num_hist, self.state_len), dtype=np.float32)
        # action history tracking
        self.action_hist = np.zeros((num_envs, num_hist, 4), dtype=np.float32)

        # Define any other environment-specific parameters
        self.max_steps = 1200  # Maximum number of steps in an episode
        self.dt = np.float32(0.01)  # Time step duration

        self.step_counts = np.zeros(num_envs, dtype=int)
        self.actions = np.zeros((num_envs, 4), dtype=np.float32)
        self.prev_actions = np.zeros((num_envs, 4), dtype=np.float32)
        self.dones = np.zeros(num_envs, dtype=bool)
        self.final_gate_passed = np.zeros(num_envs, dtype=bool)
        self.episode_distance = np.zeros(num_envs, dtype=np.float64)
        self.episode_speed_sum = np.zeros(num_envs, dtype=np.float64)
        self.episode_speed_max = np.zeros(num_envs, dtype=np.float64)
        self.episode_bank_angle_sum = np.zeros(num_envs, dtype=np.float64)
        self.episode_action_sum = np.zeros(num_envs, dtype=np.float64)
        self.episode_action_sq_sum = np.zeros(num_envs, dtype=np.float64)
        self.episode_action_count = np.zeros(num_envs, dtype=np.int64)
        self.episode_action_rpm_diff_sum = np.zeros(num_envs, dtype=np.float64)
        self.episode_metric_steps = np.zeros(num_envs, dtype=np.int64)
        self.episode_hover_steps = np.zeros(num_envs, dtype=np.int64)
        self.hover_streak = np.zeros(num_envs, dtype=np.int64)
        self.best_target_distance = np.full(num_envs, np.inf, dtype=np.float64)
        self.hover_success = np.zeros(num_envs, dtype=bool)
        self.required_hover_steps = 100
        self.hover_distance_threshold = 0.15
        self.hover_speed_threshold = 0.20
        self.hover_bank_angle_threshold = 0.20
        self.hover_rate_threshold = 0.35

        self.update_states = self.update_states_gate

        self.pause = False

    def reset_seed(self):
        if self._seed_value is not None:
            np.random.seed(self._seed_value)
            torch.manual_seed(self._seed_value)

    def update_states_gate(self):
        # Transform pos and vel in gate frame
        gate_pos = self.gate_pos[self.target_gates % self.num_gates]
        gate_yaw = self.gate_yaw[self.target_gates % self.num_gates]

        # Rotation matrix from world frame to gate frame
        R = np.array([
            [np.cos(gate_yaw), np.sin(gate_yaw)],
            [-np.sin(gate_yaw), np.cos(gate_yaw)]
        ]).transpose((2, 1, 0))

        # new state array to prevent the weird bug related to indexing ([:] syntax)
        new_states = np.zeros((self.num_envs, self.state_len), dtype=np.float32)

        # Update positions
        pos_W = self.world_states[:, 0:3]
        pos_G = (pos_W[:, np.newaxis, 0:2] - gate_pos[:, np.newaxis, 0:2]) @ R
        new_states[:, 0:2] = pos_G[:, 0, :]
        new_states[:, 2] = pos_W[:, 2] - gate_pos[:, 2]

        # Update velocities
        vel_W = self.world_states[:, 3:6]
        vel_G = (vel_W[:, np.newaxis, 0:2]) @ R
        new_states[:, 3:5] = vel_G[:, 0, :]
        new_states[:, 5] = vel_W[:, 2]

        # Update attitude
        new_states[:, 6:8] = self.world_states[:, 6:8]
        yaw = self.world_states[:, 8] - gate_yaw
        yaw %= 2 * np.pi
        yaw[yaw > np.pi] -= 2 * np.pi
        yaw[yaw < -np.pi] += 2 * np.pi
        new_states[:, 8] = yaw

        # Update rates
        new_states[:, 9:12] = self.world_states[:, 9:12]

        # Update rpms
        new_states[:, 12:16] = self.world_states[:, 12:16]

        # Update future gates relative to current gate ([0,0,0,0] for out of bounds)
        for i in range(self.gates_ahead):
            indices = (self.target_gates + i + 1)
            # loop when out of bounds
            indices = indices % self.num_gates
            valid = indices < self.num_gates
            new_states[valid, 16 + 4 * i:16 + 4 * i + 3] = self.gate_pos_rel[indices[valid]]
            new_states[valid, 16 + 4 * i + 3] = self.gate_yaw_rel[indices[valid]]

        # update action history
        self.action_hist = np.roll(self.action_hist, 1, axis=1)
        self.action_hist[:, 0] = self.actions

        for i in range(self.num_action_history):
            new_states[:, 16 + 4 * self.gates_ahead + 4 * i:16 + 4 * self.gates_ahead + 4 * i + 4] = self.action_hist[:, (i + 1) * self.history_step_size - 1]

        # update param encoding
        if self.param_input:
            new_states[:, 16 + 4 * self.gates_ahead + 4 * self.num_action_history:] = self.param_encoding

        # update state history
        self.state_hist = np.roll(self.state_hist, 1, axis=1)
        self.state_hist[:, 0] = new_states

        # print('new_states \n', new_states)

        # stack history up to self.num_state_history
        self.states = self.state_hist[:, 0:(self.num_state_history + 1) * self.history_step_size:self.history_step_size].reshape((self.num_envs, -1))
        if self.low_obs:
            self.states = np.concatenate([self.states[:, 0:3], self.states[:, 6:9], self.states[:, 12:]], axis=1)
        # print('states \n', self.states)

    def _reset_episode_metrics(self, dones):
        if not np.any(dones):
            return

        self.episode_distance[dones] = 0.0
        self.episode_speed_sum[dones] = 0.0
        self.episode_speed_max[dones] = 0.0
        self.episode_bank_angle_sum[dones] = 0.0
        self.episode_action_sum[dones] = 0.0
        self.episode_action_sq_sum[dones] = 0.0
        self.episode_action_count[dones] = 0
        self.episode_action_rpm_diff_sum[dones] = 0.0
        self.episode_metric_steps[dones] = 0
        self.episode_hover_steps[dones] = 0
        self.hover_streak[dones] = 0
        self.best_target_distance[dones] = np.inf
        self.hover_success[dones] = False

    def _motor_state_to_speed_rpm(self, motor_state):
        motor_speed_rad_s = (motor_state + 1.0) * 0.5 * (w_max_n - w_min_n) + w_min_n
        return motor_speed_rad_s * (60.0 / (2.0 * np.pi))

    def _action_to_commanded_speed_rpm(self, actions, params):
        U = (actions + 1.0) * 0.5
        k_param = params[:, -3][:, None]
        w_min_param = params[:, -2][:, None]
        w_max_param = params[:, -1][:, None]
        commanded_speed_rad_s = (w_max_param - w_min_param) * np.sqrt(
            np.clip(k_param * U ** 2 + (1.0 - k_param) * U, a_min=0.0, a_max=None)
        ) + w_min_param
        return commanded_speed_rad_s * (60.0 / (2.0 * np.pi))

    def _bank_angle_from_states(self, states):
        phi = states[:, 6]
        theta = states[:, 7]
        return np.arccos(np.clip(np.cos(phi) * np.cos(theta), -1.0, 1.0))

    def reset_(self, dones):
        num_reset = dones.sum()
        self.target_gates[dones] = 0
        target = self.gate_pos[0]

        if self.initialize_uniform or self.initialize_at_random_gates:
            x0 = target[0] + np.random.uniform(-self.start_xy_radius, self.start_xy_radius, size=(num_reset,))
            y0 = target[1] + np.random.uniform(-self.start_xy_radius, self.start_xy_radius, size=(num_reset,))
            z0 = target[2] + np.random.uniform(-self.start_z_offset, self.start_z_offset, size=(num_reset,))
        else:
            x0 = self.start_pos[0] + np.random.uniform(-self.start_pos_jitter_xy, self.start_pos_jitter_xy, size=(num_reset,))
            y0 = self.start_pos[1] + np.random.uniform(-self.start_pos_jitter_xy, self.start_pos_jitter_xy, size=(num_reset,))
            z0 = self.start_pos[2] + np.random.uniform(-self.start_pos_jitter_z, self.start_pos_jitter_z, size=(num_reset,))

        z0 = np.clip(z0, -6.0, -0.25)

        vx0 = np.random.uniform(-self.start_velocity_limit, self.start_velocity_limit, size=(num_reset,))
        vy0 = np.random.uniform(-self.start_velocity_limit, self.start_velocity_limit, size=(num_reset,))
        vz0 = np.random.uniform(-self.start_velocity_limit, self.start_velocity_limit, size=(num_reset,))

        phi0 = np.random.uniform(-self.start_tilt_limit, self.start_tilt_limit, size=(num_reset,))
        theta0 = np.random.uniform(-self.start_tilt_limit, self.start_tilt_limit, size=(num_reset,))
        psi0 = self.gate_yaw[0] + np.random.uniform(-self.start_yaw_limit, self.start_yaw_limit, size=(num_reset,))

        p0 = np.random.uniform(-self.start_rate_limit, self.start_rate_limit, size=(num_reset,))
        q0 = np.random.uniform(-self.start_rate_limit, self.start_rate_limit, size=(num_reset,))
        r0 = np.random.uniform(-self.start_rate_limit, self.start_rate_limit, size=(num_reset,))

        w10 = np.random.uniform(-self.start_motor_state_limit, self.start_motor_state_limit, size=(num_reset,))
        w20 = np.random.uniform(-self.start_motor_state_limit, self.start_motor_state_limit, size=(num_reset,))
        w30 = np.random.uniform(-self.start_motor_state_limit, self.start_motor_state_limit, size=(num_reset,))
        w40 = np.random.uniform(-self.start_motor_state_limit, self.start_motor_state_limit, size=(num_reset,))

        self.world_states[dones] = np.stack(
            [x0, y0, z0, vx0, vy0, vz0, phi0, theta0, psi0, p0, q0, r0, w10, w20, w30, w40], axis=1)

        self.step_counts[dones] = np.zeros(num_reset)
        self._reset_episode_metrics(dones)

        # update params (domain randomization)
        self.params[dones] = self.randomization(num_reset)

        # update param encoding (used for parameter input)
        if self.param_input:
            params_with_noise = self.params[dones] * np.random.uniform(1 - self.param_input_noise,
                                                                       1 + self.param_input_noise,
                                                                       size=self.params[dones].shape)
            self.param_encoding[dones] = param_encoding(params_with_noise.T).T

            # update states
        self.update_states()
        return self.states

    def reset(self):
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def step_async(self, actions):
        self.prev_actions = self.actions
        self.actions = actions

    def step_wait(self):
        self.world_states = np.asarray(self.world_states, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.params = np.asarray(self.params, dtype=np.float32)

        new_states = self.world_states + self.dt * f_func(self.world_states.T, self.actions.T, self.params.T).T

        self.step_counts += 1

        pos_old = self.world_states[:, 0:3]
        pos_new = new_states[:, 0:3]
        pos_gate = self.gate_pos[self.target_gates % self.num_gates]
        yaw_gate = self.gate_yaw[self.target_gates % self.num_gates]
        step_distance = np.linalg.norm(pos_new - pos_old, axis=1)
        step_speed = np.linalg.norm(new_states[:, 3:6], axis=1)
        step_bank_angle = self._bank_angle_from_states(new_states)
        commanded_motor_speed_rpm = self._action_to_commanded_speed_rpm(self.actions, self.params)
        actual_motor_speed_rpm = self._motor_state_to_speed_rpm(new_states[:, 12:16])
        step_action_rpm_diff = np.mean(np.abs(commanded_motor_speed_rpm - actual_motor_speed_rpm), axis=1)

        self.episode_distance += step_distance
        self.episode_speed_sum += step_speed
        self.episode_speed_max = np.maximum(self.episode_speed_max, step_speed)
        self.episode_bank_angle_sum += step_bank_angle
        self.episode_action_sum += np.sum(self.actions, axis=1)
        self.episode_action_sq_sum += np.sum(self.actions ** 2, axis=1)
        self.episode_action_count += self.actions.shape[1]
        self.episode_action_rpm_diff_sum += step_action_rpm_diff
        self.episode_metric_steps += 1

        # #### HOVER REWARDS #####
        # # Rewards
        # d2g_old = np.linalg.norm(pos_old - pos_gate, axis=1)
        # d2g_new = np.linalg.norm(pos_new - pos_gate, axis=1)
        # rat_penalty = 0.005 * np.linalg.norm(new_states[:, 9:12], axis=1)
        # angle_penalty = 0.01 *np.linalg.norm(new_states[:, 6:8], axis=1)
        # heading_penalty = 0.005 * np.abs(new_states[:, 8])
        # # action_penalty = 0.0 * np.linalg.norm((self.actions + 1) / 2, axis=1)
        # action_penalty_delta = 0.05 * np.linalg.norm((self.actions - self.prev_actions), axis=1)
        # pos_penalty = 1.0 * np.linalg.norm(pos_old - pos_new, axis=1)
        # # print("Pos penalty:", pos_penalty, "Position old:", pos_old, "Position new:", pos_new)
        # # print("Rat penalty:", rat_penalty, "Rates:", new_states[:, 9:12])
        # # print("Angle penalty:", angle_penalty, "Angles:", new_states[:, 6:8])
        # # print("Action penalty:", action_penalty_delta, "Actions:", self.actions, "Prev actions:", self.prev_actions)

        # prog_rewards = d2g_old - d2g_new
        # # max_speed = 12.0
        # # cap progress rewards to be less than max_speed*dt
        # # prog_rewards[prog_rewards > max_speed*self.dt] = max_speed*self.dt

        # # rewards = (d2g_old - d2g_new) - rat_penalty - angle_penalty
        # # rewards = prog_rewards - rat_penalty - action_penalty_delta
        # rewards = 0.5 * (- pos_penalty - rat_penalty - angle_penalty - heading_penalty - action_penalty_delta)
        # # print("Actions:", self.actions)

        # in_hover_region = (
        #     (pos_penalty < 0.02) &
        #     (angle_penalty < 0.05) &
        #     (rat_penalty < 0.05)
        # )
        # rewards[in_hover_region] += 0.1
        # # Print indeces where in hover region is true
        # # print("In hover region:", np.where(in_hover_region))

        # # Gate passing/collision
        # normal = np.array([np.cos(yaw_gate), np.sin(yaw_gate)]).T
        # # dot product of normal and position vector over axis 1
        # pos_old_projected = (pos_old[:, 0] - pos_gate[:, 0]) * normal[:, 0] + (pos_old[:, 1] - pos_gate[:, 1]) * normal[:, 1]
        # pos_new_projected = (pos_new[:, 0] - pos_gate[:, 0]) * normal[:, 0] + (pos_new[:, 1] - pos_gate[:, 1]) * normal[:, 1]
        # passed_gate_plane = (pos_old_projected < 0) & (pos_new_projected > 0)
        # gate_size = 1.5
        # gate_passed = passed_gate_plane & np.all(np.abs(pos_new - pos_gate) < gate_size / 2, axis=1)
        # gate_collision = passed_gate_plane & np.any(np.abs(pos_new - pos_gate) > gate_size / 2, axis=1)

        # # Gate reward + dist penalty
        # # rewards[gate_passed] = 1 #10 - 10*d2g_new[gate_passed]

        # # Gate collision penalty
        # # rewards[gate_collision] = -50

        # # Ground collision penalty (z > 0)
        # ground_collision = new_states[:, 2] > 0
        # rewards[ground_collision] = -50

        # # Check out of bounds
        # out_of_bounds = np.any(np.abs(new_states[:, 0:2]) > 5, axis=1)  # edges of the grid
        # out_of_bounds |= new_states[:, 2] < -7  # max height (z-axis point down)
        # out_of_bounds |= np.any(np.abs(new_states[:, 9:12]) > 1000, axis=1)  # prevent numerical issues
        # rewards[out_of_bounds] = -50

        # # Check number of steps
        # max_steps_reached = self.step_counts >= self.max_steps

        # # Update target gate
        # self.target_gates[gate_passed] += 1
        # self.target_gates[gate_passed] %= self.num_gates

        # # Check if final gate has been passed
        # self.final_gate_passed = self.target_gates >= self.num_gates

        # # give reward for passing final gate
        # # rewards[gate_passed] += 2.5
        # # rewards[self.final_gate_passed] += 50
        # count = self.step_counts % 10 == 0
        # rewards[count] += 2
        # rewards[max_steps_reached] += 100

        # # Check if the episode is done
        # dones = max_steps_reached | ground_collision | out_of_bounds #| gate_collision  # | self.final_gate_passed
        # self.dones = dones

        d2g_old = np.linalg.norm(pos_old - pos_gate, axis=1)
        d2g_new = np.linalg.norm(pos_new - pos_gate, axis=1)
        vel_penalty = 0.10 * np.linalg.norm(new_states[:, 3:6], axis=1)
        rat_penalty = 0.02 * np.linalg.norm(new_states[:, 9:12], axis=1)
        bank_angle_penalty = 0.05 * step_bank_angle
        action_penalty_delta = 0.01 * np.linalg.norm(self.actions - self.prev_actions, axis=1)

        yaw_error = new_states[:, 8] - yaw_gate
        yaw_error %= 2 * np.pi
        yaw_error[yaw_error > np.pi] -= 2 * np.pi
        yaw_error[yaw_error < -np.pi] += 2 * np.pi
        yaw_penalty = 0.02 * np.abs(yaw_error)

        prog_rewards = 1.5 * (d2g_old - d2g_new)
        pos_penalty = 0.40 * d2g_new
        rewards = prog_rewards - pos_penalty - vel_penalty - rat_penalty - bank_angle_penalty - yaw_penalty - action_penalty_delta

        in_hover_region = (
            (d2g_new < self.hover_distance_threshold) &
            (step_speed < self.hover_speed_threshold) &
            (step_bank_angle < self.hover_bank_angle_threshold) &
            (np.linalg.norm(new_states[:, 9:12], axis=1) < self.hover_rate_threshold)
        )
        rewards[in_hover_region] += 0.25

        self.episode_hover_steps += in_hover_region.astype(np.int64)
        self.hover_streak = np.where(in_hover_region, self.hover_streak + 1, 0)
        self.best_target_distance = np.minimum(self.best_target_distance, d2g_new)
        self.hover_success = self.hover_streak >= self.required_hover_steps
        rewards[self.hover_success] += 10.0

        gate_passed = np.zeros(self.num_envs, dtype=bool)
        gate_collision = np.zeros(self.num_envs, dtype=bool)

        ground_collision = new_states[:, 2] > 0
        rewards[ground_collision] = -10

        out_of_bounds = np.any(np.abs(new_states[:, 0:2] - pos_gate[:, 0:2]) > 5, axis=1)
        out_of_bounds |= np.abs(new_states[:, 2] - pos_gate[:, 2]) > 5
        out_of_bounds |= np.any(np.abs(new_states[:, 9:12]) > 1000, axis=1)
        rewards[out_of_bounds] = -10

        max_steps_reached = self.step_counts >= self.max_steps
        dones = max_steps_reached | ground_collision | out_of_bounds | self.hover_success
        self.dones = dones

        # Write info dicts
        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if dones[i]:
                metric_steps = max(int(self.episode_metric_steps[i]), 1)
                action_count = max(int(self.episode_action_count[i]), 1)
                action_mean = self.episode_action_sum[i] / action_count
                action_var = max(self.episode_action_sq_sum[i] / action_count - action_mean ** 2, 0.0)
                infos[i]["terminal_observation"] = self.states[i]
                infos[i]["TimeLimit.time"] = self.step_counts[i]
                infos[i]["distance_travelled"] = float(self.episode_distance[i])
                infos[i]["max_speed"] = float(self.episode_speed_max[i])
                infos[i]["avg_speed"] = float(self.episode_speed_sum[i] / metric_steps)
                infos[i]["avg_bank_angle"] = float(self.episode_bank_angle_sum[i] / metric_steps)
                infos[i]["avg_action_command"] = float(action_mean)
                infos[i]["action_std"] = float(np.sqrt(action_var))
                infos[i]["avg_action_rpm_diff"] = float(self.episode_action_rpm_diff_sum[i] / metric_steps)
                infos[i]["best_target_distance"] = float(self.best_target_distance[i])
                infos[i]["hover_steps"] = int(self.episode_hover_steps[i])
                infos[i]["hover_success"] = bool(self.hover_success[i])
            infos[i]["TimeLimit.truncated"] = max_steps_reached[i]
            # extra info for debugging
            infos[i]["ground_collision"] = ground_collision[i]
            infos[i]["out_of_bounds"] = out_of_bounds[i]
            infos[i]["gate_collision"] = gate_collision[i]
            infos[i]["gate_passed"] = gate_passed[i]
            infos[i]["target_distance"] = float(d2g_new[i])
            infos[i]["hover_region"] = bool(in_hover_region[i])
            infos[i]["hover_success"] = bool(self.hover_success[i])
            infos[i]["rewards"] = rewards[i]
            # infos[i]["time_step"] = self.step_counts[i]

        # Pause if collision
        if self.pause:
            dones = dones & ~dones
            self.dones = dones
        elif self.pause_if_collision:
            # dones = max_steps_reached | final_gate_passed | out_of_bounds
            update = ~dones  # ~(gate_collision | ground_collision)
            # Update world states
            self.world_states[update] = new_states[update]
            self.update_states()
            # Reset env if done (and update states)
            # self.reset_(dones)
        else:
            # Update world states
            self.world_states = new_states
            # reset env if done (and update states)
            self.reset_(dones)

        return self.states, rewards, dones, infos

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed_value = seed
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            # if hasattr(self.action_space, 'seed'):
            #     self.action_space.seed(seed)
            # if hasattr(self.observation_space, 'seed'):
            #     self.observation_space.seed(seed)

        return [seed]

    def get_attr(self, attr_name, indices=None):
        raise AttributeError()

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def render(self, mode='human'):
        # Outputs a dict containing all information for rendering
        state_dict = dict(
            zip(['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'w1', 'w2', 'w3', 'w4'],
                self.world_states.T))
        # Rescale actions to [0,1] for rendering
        action_dict = dict(zip(['u1', 'u2', 'u3', 'u4'], (np.array(self.actions.T) + 1) / 2))
        return {**state_dict, **action_dict}


# Compatibility alias so training code can swap module imports without
# renaming the environment class.
Quadcopter3DGates = Quadcopter3DHover


parallel_env = Quadcopter3DHover(
    num_envs=100,
    gates_pos=gate_pos,
    gate_yaw=gate_yaw,
    start_pos=start_pos,
    randomization=RANDOMIZATION,
    gates_ahead=1,
    num_state_history=state_history,
    num_action_history=action_history,
    history_step_size=history_step_size,
    param_input=param_input,
    param_input_noise=param_input_noise,
    initialize_uniform=True,
    initialize_at_random_gates=False,
    seed= SEED
)
test_env = Quadcopter3DHover(
    num_envs=1,
    gates_pos=gate_pos,
    gate_yaw=gate_yaw,
    start_pos=start_pos,
    randomization=RANDOMIZATION,
    gates_ahead=1,
    num_state_history=state_history,
    num_action_history=action_history,
    history_step_size=history_step_size,
    param_input=param_input,
    param_input_noise=param_input_noise,
    initialize_uniform=True,
    initialize_at_random_gates=False
)

parallel_env_low_obs = Quadcopter3DHover(
    num_envs=100,
    gates_pos=gate_pos,
    gate_yaw=gate_yaw,
    start_pos=start_pos,
    randomization=RANDOMIZATION,
    gates_ahead=1,
    num_state_history=state_history,
    num_action_history=action_history,
    history_step_size=history_step_size,
    param_input=param_input,
    param_input_noise=param_input_noise,
    initialize_uniform=True,
    initialize_at_random_gates=False,
    seed= SEED,
    low_obs=True
)
test_env_low_obs = Quadcopter3DHover(
    num_envs=1,
    gates_pos=gate_pos,
    gate_yaw=gate_yaw,
    start_pos=start_pos,
    randomization=RANDOMIZATION,
    gates_ahead=1,
    num_state_history=state_history,
    num_action_history=action_history,
    history_step_size=history_step_size,
    param_input=param_input,
    param_input_noise=param_input_noise,
    initialize_uniform=True,
    initialize_at_random_gates=False,
    low_obs=True
)

clean_env = Quadcopter3DHover(
    num_envs=1,
    gates_pos=gate_pos,
    gate_yaw=gate_yaw,
    start_pos=start_pos,
    randomization=RANDOMIZATION,
    gates_ahead=1,
    num_state_history=state_history,
    num_action_history=action_history,
    history_step_size=history_step_size,
    param_input=param_input,
    param_input_noise=param_input_noise,
    initialize_uniform = True,
    initialize_at_random_gates = False,
    seed = SEED
)
