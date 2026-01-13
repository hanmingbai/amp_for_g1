import glob
from legged_gym.envs.base.legged_robot_amp_config import LeggedRobotAmpCfg, LeggedRobotAmpCfgPPO

MOTION_FILES = glob.glob('datasets/G1MotionData/*')

class G1AMPCfg( LeggedRobotAmpCfg ):

    class env( LeggedRobotAmpCfg.env ):
        num_envs = 4096
        include_history_steps = None  # Number of steps of history to include.
        num_observations = 96
        num_privileged_obs = 99
        num_actions = 29
        episode_length_s = 20

        reference_state_initialization = False
        reference_state_initialization_prob = 0.85

        terminate_on_velocity = True
        terminate_on_height = True
        terminate_on_gravity = True

        amp_motion_files = MOTION_FILES


    class init_state( LeggedRobotAmpCfg.init_state ):
        pos = [0, 0, 0.8]
        default_joint_angles = {
            'left_hip_yaw_joint' : 0. ,   
            'left_hip_roll_joint' : 0,               
            'left_hip_pitch_joint' : -0.1,         
            'left_knee_joint' : 0.3,       
            'left_ankle_pitch_joint' : -0.2,     
            'left_ankle_roll_joint' : 0,     
            'right_hip_yaw_joint' : 0., 
            'right_hip_roll_joint' : 0, 
            'right_hip_pitch_joint' : -0.1,                                       
            'right_knee_joint' : 0.3,                                             
            'right_ankle_pitch_joint': -0.2,                              
            'right_ankle_roll_joint' : 0,  

            'torso_joint' : 0.,
            
            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,
            
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.4,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 1.2,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': -0.4,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 1.2,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
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
                     'waist': 150,
                     'shoulder': 40,
                     'elbow': 40,
                     'wrist': 20,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'waist': 4,
                     'shoulder': 1,
                     'elbow': 1,
                     'wrist': 0.5,
                     }   # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain( LeggedRobotAmpCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class asset( LeggedRobotAmpCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_rev_1_0.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee", "shoulder", "elbow", "torso_link", "wrist"]
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
        class scales( LeggedRobotAmpCfg.rewards.scales ):
            # *** Task *** #
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0

            # *** Penalties *** #
            orientation = -1.0
            # lin_vel_z = -0.2
            # ang_vel_xy = -0.05
            # dof_torques = -2.0e-6
            # dof_acc = -1.0e-7
            action_rate = -0.0005
            # dof_pos_limits = -10.0

            # deviation_hip_roll = -0.1
            # deviation_hip_yaw = -0.1
            # deviation_waist = -0.1
            # deviation_shoulder_roll = -0.2
            # deviation_shoulder_pitch = -0.1
            # deviation_shoulder_yaw = -0.2
            # deviation_elbow = -0.1
            deviation_wrist_roll = -0.1
            deviation_wrist_pitch = -0.1
            deviation_wrist_yaw = -0.1

            # feet_air_time = 0.5
            # feet_slide = -0.1

            # *** Termination *** #
            termination = -200

# 改进点：（1）数据增强，obs和action中数据镜像；（2）amp_obs用多帧数据；（3）

# 记录0112: 数据集没有滤波, 训练出来的策略会疯狂抖动

    class commands( LeggedRobotAmpCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [ -0.5, 3.0] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

class G1AMPCfgPPO( LeggedRobotAmpCfgPPO ):
    runner_class_name = 'AMPOnPolicyRunner'

    class policy( LeggedRobotAmpCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
    class algorithm( LeggedRobotAmpCfgPPO.algorithm ):
        amp_replay_buffer_size = 1000000

        value_loss_coef = 1.0 # 2.5 # 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-4 # 1.e-3 #5.e-4 # 1.e-3
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner( LeggedRobotAmpCfgPPO.runner ):
        run_name = 'walk'
        experiment_name = 'amp23dof'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 10000 # number of policy updates
        save_interval = 100

        amp_reward_coef = 5.0 # 5.0 # 0.3 # 0.3 # 2.5 # 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.4 # 0.4 0.8 # 0.8 # 0.6 # 0.3 # 0.6
        amp_discr_hidden_dims = [1024, 512]
        min_normalized_std = [0.05] * 29

# Motor ID 0: left_hip_pitch_joint
# Motor ID 1: left_hip_roll_joint
# Motor ID 2: left_hip_yaw_joint
# Motor ID 3: left_knee_joint
# Motor ID 4: left_ankle_pitch_joint
# Motor ID 5: left_ankle_roll_joint
# Motor ID 6: right_hip_pitch_joint
# Motor ID 7: right_hip_roll_joint
# Motor ID 8: right_hip_yaw_joint
# Motor ID 9: right_knee_joint
# Motor ID 10: right_ankle_pitch_joint
# Motor ID 11: right_ankle_roll_joint
# Motor ID 12: waist_yaw_joint
# Motor ID 13: waist_roll_joint
# Motor ID 14: waist_pitch_joint
# Motor ID 15: left_shoulder_pitch_joint
# Motor ID 16: left_shoulder_roll_joint
# Motor ID 17: left_shoulder_yaw_joint
# Motor ID 18: left_elbow_joint
# Motor ID 19: left_wrist_roll_joint
# Motor ID 20: left_wrist_pitch_joint
# Motor ID 21: left_wrist_yaw_joint
# Motor ID 22: right_shoulder_pitch_joint
# Motor ID 23: right_shoulder_roll_joint
# Motor ID 24: right_shoulder_yaw_joint
# Motor ID 25: right_elbow_joint
# Motor ID 26: right_wrist_roll_joint
# Motor ID 27: right_wrist_pitch_joint
# Motor ID 28: right_wrist_yaw_joint