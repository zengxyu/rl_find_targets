import os

from src.agent.agent_ppo_lstm import Agent
from src.network.network_ppo_lstm import PPO_LSTM2, PPO_LSTM3, PPO_LSTM

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pygame
import pygame as pg
import torch
from torch.utils.tensorboard import SummaryWriter

from src.env.config import FIELD, ACTION, MOVEMENTS
from src.env.grid_env import GridEnv
from src.utils.pygame_painter import Painter

parser = argparse.ArgumentParser()
parser.add_argument("--headless", default=False, action="store_true", help="Run in headless mode")
args = parser.parse_args()

params = {
    'name': 'ppo_lstm',
    # field
    'w': FIELD.w,
    'h': FIELD.h,
    'start_direction': MOVEMENTS[np.random.randint(0, len(ACTION))],
    'start_pos': np.random.randint(0, min(FIELD.h, FIELD.w), size=(2,)),

    'field_data': FIELD.data,
    # model params
    'update_every': 10,
    'eps_start': 0.15,  # Default/starting value of eps
    'eps_decay': 0.99999,  # Epsilon decay rate
    'eps_min': 0.15,  # Minimum epsilon
    'gamma': 0.98,
    'buffer_size': 200000,
    'batch_size': 16,
    'seq_len': 4,
    'num_layers': 1,
    'lr': 1e-4,
    'model': PPO_LSTM,  # PPO_LSTM, PPO_LSTM2, PPO_LSTM3
    'action_size': len(ACTION),

    'is_double': False,
    'is_priority_buffer': True,

    # grid params
    'max_step': 200,

    # train params
    'visualise': False,
    'is_normalize': False,
    'num_episodes': 5000000,
    'scale': 15,

    # folder params

    # output
    'output_folder': "output_lstm",
    'log_folder': 'log',
    'model_folder': 'model',
    'memory_config_dir': "memory_config",

    'use_cuda': True
}

params['log_folder'] = os.path.join(params['output_folder'], params['log_folder'])
params['model_folder'] = os.path.join(params['output_folder'], params['model_folder'])
if not os.path.exists(params['log_folder']):
    os.makedirs(params['log_folder'])
if not os.path.exists(params['model_folder']):
    os.makedirs(params['model_folder'])

painter = Painter(params) if params['visualise'] else None
grid_env = GridEnv(params, painter)

train_device = 'cuda' if torch.cuda.is_available() and params['use_cuda'] else 'cpu'

writer = SummaryWriter(log_dir=params['log_folder'])
player = Agent(params=params, writer=writer, train_agent=True, is_resume=False,
               filepath=None, train_device=torch.device(train_device))

total_rewards, smoothed_rewards = [], []

global_step = 0
T_horizon = params['seq_len']
batch_size = params["batch_size"]
all_mean_rewards = []
all_mean_losses = []

for i_episode in range(0, params['num_episodes']):
    print("\nepisode = ", i_episode)

    h_out = torch.zeros([params['num_layers'], 1, 32], dtype=torch.float)
    c_out = torch.zeros([params['num_layers'], 1, 32], dtype=torch.float)

    observed_map, robot_pose = grid_env.reset()
    done = False
    rewards = []
    losses = []
    while not done:
        # print("\nt horizon = ", t)
        global_step += 1
        h_in, c_in = h_out, c_out
        action, value, probs, h_out, c_out = player.act(observed_map, robot_pose, h_in, c_in)

        observed_map_prime, robot_pose_prime, reward, done = grid_env.step(action.detach().cpu().numpy()[0][0])

        player.store_data(
            [observed_map, robot_pose, action.detach().cpu().numpy().squeeze(), reward, observed_map_prime,
             robot_pose_prime, value.detach().cpu().numpy().squeeze(), probs.detach().cpu().numpy().squeeze(),
             done, h_in.detach().cpu().numpy(), c_in.detach().cpu().numpy(), h_out.detach().cpu().numpy(),
             c_out.detach().cpu().numpy()])

        observed_map = observed_map_prime.copy()
        robot_pose = robot_pose_prime.copy()

        rewards.append(reward)

        if params['visualise']:
            painter.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        if done:
            all_mean_rewards.append(np.mean(rewards))
            all_mean_losses.append(np.mean(losses))
            print(
                "\ni episode:{}; mean reward:{}; num_found_free_cell:{}/{};num_found_targets:{}/{}; num_found_occupied:{}/{}"
                    .format(i_episode, np.mean(rewards), grid_env.count_found_free, grid_env.count_free,
                            grid_env.count_found_target, grid_env.count_target,
                            grid_env.count_found_occupied, grid_env.count_occupied))
            writer.add_scalar('train/losses_smoothed', np.mean(all_mean_losses[max(0, i_episode - 200):]),
                              i_episode)
            writer.add_scalar('train/loss_per_episode', np.mean(losses), i_episode)
            writer.add_scalar('train/rewards_smoothed', np.mean(all_mean_rewards[max(0, i_episode - 200):]),
                              i_episode)
            writer.add_scalar("train/reward_per_episode", np.mean(rewards), i_episode)
            writer.add_scalar('train/num_found_free_cell', grid_env.count_found_free, i_episode)
            writer.add_scalar('train/num_found_targets', grid_env.count_found_target, i_episode)
            writer.add_scalar('train/num_found_total_cell',
                              grid_env.count_found_free + grid_env.count_found_target, i_episode)

            break

        if player.memory.is_full_batch():
            loss = player.train_net()

            player.scheduler.step()
            player.memory.reset_data()

            losses.append(loss)
            writer.add_scalar('train/loss', loss, global_step)

# save dict
player.store_model("Agent_ppo_state_dict.mdl")
plt.plot(total_rewards)
plt.plot(smoothed_rewards)
plt.title("Total reward per episode")
plt.savefig("rewards.png")

if not args.headless:
    pg.quit()
