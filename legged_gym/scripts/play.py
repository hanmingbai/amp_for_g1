import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.reference_state_initialization = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 100

    for i in range(5*int(env.max_episode_length)):
        actions = policy(obs.detach())
        env.commands[robot_index, 0] = 0.6
        env.commands[robot_index, 1] = 0.0
        env.commands[robot_index, 2] = 0.0
        obs, _, rews, dones, infos = env.step(actions.detach())[:5]

        if i < stop_state_log:
            logger.log_states(
                {
                    'command_x':env.commands[robot_index, 0].item(),
                    'command_y':env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'left_knee_torque': env.torques[robot_index, 3].item(),
                    'right_knee_torque':env.torques[robot_index, 9].item(),
                    'contact_forces':env.contact_forces[robot_index,env.feet_indices, 2].cpu().numpy(),

                    'left_hip_pitch': env.torques[robot_index, 0].item(),
                    'right_hip_pitch':env.torques[robot_index, 6].item(),

                    'left_hip_roll': env.torques[robot_index, 1].item(),
                    'right_hip_roll':env.torques[robot_index, 7].item(),

                    'left_hip_yaw': env.torques[robot_index, 2].item(),
                    'right_hip_yaw':env.torques[robot_index, 8].item(),

                    'left_ankle_pitch': env.torques[robot_index, 4].item(),
                    'right_ankle_pitch':env.torques[robot_index, 10].item(),

                    'left_ankle_roll': env.torques[robot_index, 5].item(),
                    'right_ankle_roll':env.torques[robot_index, 11].item(),
                }
            )
        elif i==stop_state_log:
            logger.plot_states()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
