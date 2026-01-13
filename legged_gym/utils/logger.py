import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def plot_states(self):
        self.plot_process = Process(target=self._plot_torques)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log

        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]!=[]: a.plot(time, log["command_x"], label='command_x')
        # a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='command_x')
        a.legend()
        # plot contact forces
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]!=[]: a.plot(time, log["command_y"], label='command_y')
        # a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='command_x')
        a.legend()
        a = axs[0, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]!=[]: a.plot(time, log["command_yaw"], label='command_yaw')
        # a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='command_x')
        a.legend()

        # plot torques
        a = axs[1, 0]
        if log["left_knee_torque"]!=[]: a.plot(time, log["left_knee_torque"], label='left_knee_torque')
        # a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        a = axs[1, 1]
        if log["right_knee_torque"]!=[]: a.plot(time, log["right_knee_torque"], label='right_knee_torque')
        # a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        a = axs[1, 2]
        if log["contact_forces"]!=[]: a.plot(time, log["contact_forces"], label='contact_forces')
        # a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()

        plt.show()

    def _plot_torques(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log

        a = axs[0, 0]
        if log["left_hip_pitch"]!=[]: a.plot(time, log["left_hip_pitch"], label='left_hip_pitch')
        if log["right_hip_pitch"]!=[]: a.plot(time, log["right_hip_pitch"], label='right_hip_pitch')
        a.legend()

        a = axs[0, 1]
        if log["left_hip_roll"]!=[]: a.plot(time, log["left_hip_roll"], label='left_hip_roll')
        if log["right_hip_roll"]!=[]: a.plot(time, log["right_hip_roll"], label='right_hip_roll')
        a.legend()

        a = axs[0, 2]
        if log["left_hip_yaw"]!=[]: a.plot(time, log["left_hip_yaw"], label='left_hip_yaw')
        if log["right_hip_yaw"]!=[]: a.plot(time, log["right_hip_yaw"], label='right_hip_yaw')
        a.legend()

        a = axs[1, 0]
        if log["left_knee_torque"]!=[]: a.plot(time, log["left_knee_torque"], label='left_knee_torque')
        if log["right_knee_torque"]!=[]: a.plot(time, log["right_knee_torque"], label='right_knee_torque')
        a.legend()

        a = axs[1, 1]
        if log["left_ankle_pitch"]!=[]: a.plot(time, log["left_ankle_pitch"], label='left_ankle_pitch')
        if log["right_ankle_pitch"]!=[]: a.plot(time, log["right_ankle_pitch"], label='right_ankle_pitch')
        a.legend()

        a = axs[1, 2]
        if log["left_ankle_roll"]!=[]: a.plot(time, log["left_ankle_roll"], label='left_ankle_roll')
        if log["right_ankle_roll"]!=[]: a.plot(time, log["right_ankle_roll"], label='right_ankle_roll')
        a.legend()

        plt.show()

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()   

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()