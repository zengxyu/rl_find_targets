import os

import pygame
from torch.utils.tensorboard import SummaryWriter

from src.agent.agent_q_learning import Agent
from src.env.grid_env import GridEnv
import numpy as np

params = {
    'name': 'q_learning',
    # model params
    'eps': 0.2,  # epsilon
    'gamma': 0.8,
    'alpha': 0.2,
    # grid params
    'max_step': 500,

    # train params
    'num_episodes': 5000000,
    'visualise': False,
    'scale': 10,

    # folder params
    # data
    'field_file_path': "data/ori_data.txt",
    # output
    'output_folder': "output_q_learning",
    'log_folder': 'log',
    'model_folder': 'model',
}

params['log_folder'] = os.path.join(params['output_folder'], params['log_folder'])
params['model_folder'] = os.path.join(params['output_folder'], params['model_folder'])
if not os.path.exists(params['log_folder']):
    os.makedirs(params['log_folder'])
if not os.path.exists(params['model_folder']):
    os.makedirs(params['model_folder'])

writer = SummaryWriter(log_dir=params['log_folder'])
grid_env = GridEnv(params)

model_path = os.path.join("", "")
q_learning_agent = Agent(grid_env.h, grid_env.w, params)

if __name__ == '__main__':
    all_mean_rewards = []
    for i_episode in range(params['num_episodes']):
        observed_map, robot_pose = grid_env.reset()
        done = False

        rewards = []

        while not done:
            action = q_learning_agent.act_eps_greedy(robot_pose, params['eps'])

            key1 = "{}_{}_{}".format(robot_pose[0], robot_pose[1], action)

            observed_map_next, robot_pose_next, reward, done = grid_env.step(action)

            # s1处的最大动作
            action_next = q_learning_agent.act_greedy(robot_pose_next)

            key2 = "{}_{}_{}".format(robot_pose_next[0], robot_pose_next[1], action_next)

            q_learning_agent.learn(key1, key2, reward)

            # 转到下一个状态
            robot_pose = robot_pose_next

            if params['visualise']:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()

            rewards.append(reward)

            if done:
                if (i_episode + 1) % 200 == 0:
                    # plt.cla()
                    model_save_path = os.path.join(params['model_folder'],
                                                   "Agent_dqn_state_dict_%d.mdl" % (i_episode + 1))
                    q_learning_agent.store_model(model_save_path)

                all_mean_rewards.append(np.mean(rewards))
                print()
                print("i episode:{}; mean reward:{}; num_found_free_cell:{};num_found_targets:{}"
                      .format(i_episode, np.mean(rewards), grid_env.count_found_free, grid_env.count_found_target))
                writer.add_scalar('train/smoothed_rewards', np.mean(all_mean_rewards[max(0, i_episode - 200):]),
                                  i_episode)
                writer.add_scalar("train/reward_per_episode", np.mean(rewards), i_episode)
                writer.add_scalar('train/num_found_free_cell', grid_env.count_found_free, i_episode)
                writer.add_scalar('train/num_found_targets', grid_env.count_found_target, i_episode)
                writer.add_scalar('train/num_found_total_cell', grid_env.count_found_free + grid_env.count_found_target,
                                  i_episode)
