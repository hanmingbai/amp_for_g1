# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import glob
import json

import numpy as np
import torch


class AMPLoader:

    # 关节数量
    NUM_ACTIONS = 12 # 12, 23, 29

    # size
    POS_SIZE = 3
    ROT_SIZE = 4

    LINVEL_SIZE = 3
    ANGVEL_SIZE = 3

    JOINT_POS_SIZE = NUM_ACTIONS # 29 # 23 # 12
    JOINT_VEL_SIZE = NUM_ACTIONS # 29 # 23 # 12
    
    if NUM_ACTIONS == 12:
        END_EFFECTOR_POS_SIZE = 6 # ankle_roll_link
    else:
        END_EFFECTOR_POS_SIZE = 18 # ankle_roll_link, shoulder_roll_link, wrist_yaw_link

    GRAVIRY_VEC_SIZE = 3

    # root pose
    POS_START_IDX = 0
    POS_END_IDX = POS_START_IDX + POS_SIZE

    ROT_START_IDX = POS_END_IDX
    ROT_END_IDX = ROT_START_IDX + ROT_SIZE

    # root vel
    LINVEL_START_IDX = ROT_END_IDX
    LINVEL_END_IDX = LINVEL_START_IDX + LINVEL_SIZE

    ANGVEL_START_IDX = LINVEL_END_IDX
    ANGVEL_END_IDX =ANGVEL_START_IDX + ANGVEL_SIZE

    # joint pos
    JOINT_POS_START_IDX = ANGVEL_END_IDX
    JOINT_POS_END_IDX = JOINT_POS_START_IDX + JOINT_POS_SIZE

    # joint vel
    JOINT_VEL_START_IDX = JOINT_POS_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    # ee pos
    END_POS_START_IDX = JOINT_VEL_END_IDX
    END_POS_END_IDX = END_POS_START_IDX + END_EFFECTOR_POS_SIZE

    # projected gravity vector
    GRAVITY_VEC_START_IDX = END_POS_END_IDX
    GRAVITY_VEC_END_IDX = GRAVITY_VEC_START_IDX + GRAVIRY_VEC_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/g1_expert_motions/*"),
        num_actions=None,
    ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        if num_actions is not None:
            # self.NUM_ACTIONS = num_actions
            print(f"num_action:{num_actions}")

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                # Remove first 7 observation dimensions (root_pos and root_orn).
                self.trajectories.append(
                    torch.tensor(motion_data[:, AMPLoader.ANGVEL_START_IDX: AMPLoader.GRAVITY_VEC_END_IDX], dtype=torch.float32, device=device)
                )
                self.trajectories_full.append(
                    torch.tensor(motion_data[:, AMPLoader.POS_START_IDX: AMPLoader.GRAVITY_VEC_END_IDX], dtype=torch.float32, device=device)
                )

                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                print(f"traj_len:{traj_len}")
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]

        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst

        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, frame1, frame2, blend):
        return (1.0 - blend) * frame1 + blend * frame2

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)
        all_frame_amp_starts = torch.zeros(
            len(traj_idxs), AMPLoader.GRAVITY_VEC_END_IDX - AMPLoader.POS_START_IDX, device=self.device
        )
        all_frame_amp_ends = torch.zeros(
            len(traj_idxs), AMPLoader.GRAVITY_VEC_END_IDX - AMPLoader.POS_START_IDX, device=self.device
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, AMPLoader.POS_START_IDX : AMPLoader.GRAVITY_VEC_END_IDX
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, AMPLoader.POS_START_IDX : AMPLoader.GRAVITY_VEC_END_IDX
            ]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        joints0, joints1 = AMPLoader.get_joint_pose(frame0), AMPLoader.get_joint_pose(frame1)
        joint_vel_0, joint_vel_1 = AMPLoader.get_joint_vel(frame0), AMPLoader.get_joint_vel(frame1)

        blend_joint_q = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([blend_joint_q, blend_joints_vel])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)

                # 基座速度与角速度
                s_vel = self.preloaded_s[idxs, AMPLoader.ANGVEL_START_IDX : AMPLoader.ANGVEL_END_IDX]
                s_vel_next = self.preloaded_s_next[idxs, AMPLoader.ANGVEL_START_IDX : AMPLoader.ANGVEL_END_IDX]
                
                # 关节位置
                s_joint_pos = self.preloaded_s[idxs, AMPLoader.JOINT_POS_START_IDX : AMPLoader.JOINT_POS_END_IDX]
                s_joint_pos_next = self.preloaded_s_next[idxs, AMPLoader.JOINT_POS_START_IDX : AMPLoader.JOINT_POS_END_IDX]

                # 关节速度
                s_joint_vel= self.preloaded_s[idxs, AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]
                s_joint_vel_next = self.preloaded_s_next[idxs, AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

                # 末端位置
                s_ee = self.preloaded_s[idxs, AMPLoader.END_POS_START_IDX : AMPLoader.END_POS_END_IDX]
                s_ee_next = self.preloaded_s_next[idxs, AMPLoader.END_POS_START_IDX : AMPLoader.END_POS_END_IDX]

                # 重力分量
                s_gravity = self.preloaded_s[idxs, AMPLoader.GRAVITY_VEC_START_IDX : AMPLoader.GRAVITY_VEC_END_IDX]
                s_gravity_next = self.preloaded_s_next[idxs, AMPLoader.GRAVITY_VEC_START_IDX : AMPLoader.GRAVITY_VEC_END_IDX]

                # 高度
                s_height = self.preloaded_s[idxs, AMPLoader.POS_START_IDX + 2:AMPLoader.POS_START_IDX + 3]
                s_height_next = self.preloaded_s_next[idxs, AMPLoader.POS_START_IDX + 2:AMPLoader.POS_START_IDX + 3]

                # amp expert obs
                # joint_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
                if AMPLoader.NUM_ACTIONS == 12:
                    joint_index = [0,1,2,3,4,5,6,7,8,9,10,11]
                    s = torch.cat([s_joint_pos[:,joint_index], s_joint_vel[:,joint_index], s_ee, s_height], dim=-1)
                    s_next = torch.cat([s_joint_pos_next[:,joint_index], s_joint_vel_next[:,joint_index], s_ee_next, s_height_next], dim=-1)
                elif AMPLoader.NUM_ACTIONS == 23:
                    joint_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
                    s_ee = self.preloaded_s[idxs, AMPLoader.END_POS_START_IDX + 6 : AMPLoader.END_POS_END_IDX]
                    s_ee_next = self.preloaded_s_next[idxs, AMPLoader.END_POS_START_IDX + 6 : AMPLoader.END_POS_END_IDX]
                    s = torch.cat([s_joint_pos[:,joint_index], s_joint_vel[:,joint_index], s_ee], dim=-1)
                    s_next = torch.cat([s_joint_pos_next[:,joint_index], s_joint_vel_next[:,joint_index], s_ee_next], dim=-1)
                else:
                    joint_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
                    s = torch.cat([s_joint_pos[:,joint_index], s_joint_vel[:,joint_index], s_ee, s_height], dim=-1)
                    s_next = torch.cat([s_joint_pos_next[:,joint_index], s_joint_vel_next[:,joint_index], s_ee_next, s_height_next], dim=-1)
                
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames))

                s = torch.vstack(s)                    
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        if AMPLoader.NUM_ACTIONS == 23:
            #           线速度                角速度                关节角度                关节速度               末端位置                          重力投影              高度
            return self.LINVEL_SIZE*0 + self.ANGVEL_SIZE*0 + self.JOINT_POS_SIZE + self.JOINT_VEL_SIZE + self.END_EFFECTOR_POS_SIZE - 6 + self.GRAVIRY_VEC_SIZE*0 + 1*0
        else:
            #           线速度                角速度                关节角度                关节速度               末端位置                          重力投影              高度
            return self.LINVEL_SIZE*0 + self.ANGVEL_SIZE*0 + self.JOINT_POS_SIZE + self.JOINT_VEL_SIZE + self.END_EFFECTOR_POS_SIZE + self.GRAVIRY_VEC_SIZE*0 + 1

    @property
    def num_motions(self):
        return len(self.trajectory_names)
    
    def get_root_pos(pose):
        return pose[AMPLoader.POS_START_IDX:AMPLoader.POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, AMPLoader.POS_START_IDX:AMPLoader.POS_END_IDX]

    def get_root_rot(pose):
        return pose[AMPLoader.ROT_START_IDX:AMPLoader.ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, AMPLoader.ROT_START_IDX:AMPLoader.ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[AMPLoader.JOINT_POS_START_IDX : AMPLoader.JOINT_POS_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, AMPLoader.JOINT_POS_START_IDX : AMPLoader.JOINT_POS_END_IDX]
    
    def get_linear_vel(pose):
        return pose[AMPLoader.LINVEL_START_IDX:AMPLoader.LINVEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, AMPLoader.LINVEL_START_IDX:AMPLoader.LINVEL_END_IDX]

    def get_angular_vel(pose):
        return pose[AMPLoader.ANGVEL_START_IDX:AMPLoader.ANGVEL_END_IDX]  

    def get_angular_vel_batch(poses):
        return poses[:, AMPLoader.ANGVEL_START_IDX:AMPLoader.ANGVEL_END_IDX]  

    def get_joint_vel(pose):
        return pose[AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    def get_end_pos(pose):
        return pose[AMPLoader.END_POS_START_IDX : AMPLoader.END_POS_END_IDX]

    def get_end_pos_batch(poses):
        return poses[:, AMPLoader.END_POS_START_IDX : AMPLoader.END_POS_END_IDX]

