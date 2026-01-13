# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import glob

from legged_gym.envs.base.legged_robot_amp_config import LeggedRobotAmpCfg, LeggedRobotAmpCfgPPO

MOTION_FILES = glob.glob('datasets/g1_23dof_expert_motions/*')


class G1AMP23DOFCfg( LeggedRobotAmpCfg ):

    class env( LeggedRobotAmpCfg.env ):
        num_envs = 4096
        include_history_steps = None  # Number of steps of history to include.
        num_observations = 78
        num_privileged_obs = 81
        num_actions = 23
        episode_length_s = 20

        reference_state_initialization = True
        reference_state_initialization_prob = 0.85

        terminate_on_velocity = True
        terminate_on_height = True
        terminate_on_gravity = True

        amp_motion_files = MOTION_FILES


    class init_state( LeggedRobotAmpCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : 0.,         
           'left_knee_joint' : 0.,       
           'left_ankle_pitch_joint' : 0.,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : 0.,                                       
           'right_knee_joint' : 0.,                                             
           'right_ankle_pitch_joint': 0.,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.,
           'waist_yaw_joint': 0.,
           'left_shoulder_pitch_joint': 0.,
           'left_shoulder_roll_joint': 0.,
           'left_shoulder_yaw_joint': 0.,
           'left_elbow_joint': 0.,
           'left_wrist_roll_joint': 0.,
           'right_shoulder_pitch_joint': 0.,
           'right_shoulder_roll_joint': 0.,
           'right_shoulder_yaw_joint': 0.,
           'right_elbow_joint': 0.,
           'right_wrist_roll_joint': 0.
        }

    class control( LeggedRobotAmpCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     'waist':40,
                     'shoulder': 40,
                     'elbow': 40,
                     'wrist': 20
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'waist':2,
                     'shoulder': 2,
                     'elbow': 2,
                     'wrist': 1
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain( LeggedRobotAmpCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class asset( LeggedRobotAmpCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_lock_23dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis","shoulder", "elbow", "knee", "hip", "torso_link"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class domain_rand( LeggedRobotAmpCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.25, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]
        randomize_phase = False

    class noise( LeggedRobotAmpCfg.noise ):
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales( LeggedRobotAmpCfg.noise.noise_scales ):
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0.1

    class rewards( LeggedRobotAmpCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        period = 0.8
        offset = 0.5
        is_stance = 0.55
        feet_swing_height = 0.08
        tracking_sigma = 0.5 # default: 0.25

        class scales( LeggedRobotAmpCfg.rewards.scales ):

            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            feet_air_time = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_acc = -1e-8
            dof_vel = -1e-5
            dof_torques = -1e-5
            action_rate = -0.001
            dof_pos_limits = -2.0
            contact_no_vel = -0.2
            termination = -200.0

            ## penalties
            # dof torques = -2.0e-6
            # feet_air_time = 0.125
            # deviation_hip_roll = -0.1
            # deviation_hip_yaw = -0.1
            # deviation_waist_yaw = -0.1
            # deviation_shoulder_roll = -0.05
            # deviation_shoulder_pitch = -0.05
            # deviation_shoulder_yaw = -0.05
            # deviation_elbow = -0.05
            # deviation_wrist_roll = -0.05
            # feet_air_time = 0.5



# 改进点：（1）数据增强，obs和action中数据镜像；（2）amp_obs用多帧数据；（3）

    class commands( LeggedRobotAmpCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [ -0.5, 2.5] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

class G1AMP23DOFCfgPPO( LeggedRobotAmpCfgPPO ):
    runner_class_name = 'AMPOnPolicyRunner'

    class policy( LeggedRobotAmpCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
    class algorithm( LeggedRobotAmpCfgPPO.algorithm ):
        amp_replay_buffer_size = 1000000

        value_loss_coef = 1.0   # 2.5 # 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4    # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-4   # 1.e-3 #5.e-4 # 1.e-3
        schedule = 'adaptive'   # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner( LeggedRobotAmpCfgPPO.runner ):
        run_name = 'walk2run'
        experiment_name = 'g1_amp_23dof'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 5000   # number of policy updates
        save_interval = 100

        amp_reward_coef = 1.0   # 0.3 # 0.3 # 2.5 # 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.5
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05] * 23

# 可以要toe的数据，可以不用； 
# 数据集可能不够好（但只需要一个完整周期就行）；
# amp崩塌处理：真假数据某一方收敛太快，调梯度惩罚；
# stage1: 不加nosie和domain_rand, stage2: 加noise和domain_rand 各训练几千轮？

# 1209: 有个奇怪的现象，以前0.25-0.6是可以训练出来的，现在0.3-0.6都训练不出来了，不知道是不是lamda=10的问题，现在用lamda=15以及训练过的lamda=10作对比实验
# 只有 tracking_lin_vel = 1.0 * 1. / (.005 * 4), tracking_ang_vel = 0.5 * 1. / (.005 * 4) 和 style rew，amp_reward_coef=2.0, amp_task_reward_lerp=0.3 勉强可以训练出来
# 0.25, 0.6 可以训练出行走的，但是风格步态不怎么自然 joint pos. joint vel, gravity vec, z pos
# lamda=15, coef=0.3, lerp=0.6
