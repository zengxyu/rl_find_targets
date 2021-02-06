import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from torch.utils.tensorboard import SummaryWriter

from src.env.grid_env import GridEnv
from src.agent import Agent
import numpy as np
import pygame

visualise = True

grid_env = GridEnv(visualize=visualise, field_file_path="src/data/ori_data.txt", max_step=500)

BATCH_SIZE = 256
GAMMA = 0.8
EPS_START = 0.5
EPS_END = 0.8
EPS_DECAY = 200
TARGET_UPDATE = 10

num_episodes = 5000000

best = dict()

dqn_agent = Agent()

writer = SummaryWriter(log_dir=os.path.join("log"))

output_model_dir = "model"
if not os.path.exists(output_model_dir):
    os.makedirs(output_model_dir)

all_mean_rewards = []

for i_episode in range(num_episodes):
    observed_map, robot_pose = grid_env.reset()
    done = False
    rewards = []

    while not done:
        action = dqn_agent.act(observed_map, robot_pose)
        observed_map_next, robot_pose_next, reward, done = grid_env.step(action)
        dqn_agent.step(state=[observed_map, robot_pose], action=action, reward=reward,
                       next_state=[observed_map_next, robot_pose_next], done=done)
        # print("action=", action, ";reward:", reward, ";done:", done)
        if visualise:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
        rewards.append(reward)
        if done:
            if (i_episode + 1) % 10000 == 0:
                # plt.cla()
                model_save_path = os.path.join(output_model_dir, "Agent_dqn_state_dict_%d.mdl" % (i_episode + 1))
                dqn_agent.store_model(model_save_path)

            all_mean_rewards.append(np.mean(rewards))
            print()
            print("i episode:{}; mean reward:{}; num_found_free_cell:{};num_found_targets:{}"
                  .format(i_episode, np.mean(rewards), grid_env.count_found_free, grid_env.count_found_target))
            writer.add_scalar('train/smoothed_rewards', np.mean(all_mean_rewards[max(0, i_episode - 50):]), i_episode)
            writer.add_scalar("train/reward_per_episode", np.mean(rewards), i_episode)
            writer.add_scalar('train/num_found_free_cell', grid_env.count_found_free, i_episode)
            writer.add_scalar('train/num_found_targets', grid_env.count_found_target, i_episode)
            writer.add_scalar('train/num_found_total_cell', grid_env.count_found_free + grid_env.count_found_target,
                              i_episode)

print('Complete')
