import sys
import os

from src.agent.agent_dqn import Agent

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from torch.utils.tensorboard import SummaryWriter

from src.env.grid_env import GridEnv
import pygame
from src.env.config import *
from src.utils.pygame_painter import Painter

params = {
    'name': 'dqn',
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
    'gamma': 0.9,
    'buffer_size': 200000,
    'batch_size': 128,
    'action_size': len(ACTION),

    'is_double': False,
    'is_priority_buffer': True,

    # grid params
    'max_step': 200,

    # train params
    'is_train': True,
    'visualise': False,
    'is_normalize': False,
    'num_episodes': 5000000,
    'scale': 15,

    # folder params

    # output
    'output_folder': "output_dqn3",
    'log_folder': 'log',
    'model_folder': 'model',
    'memory_config_dir': "memory_config"

}

params['log_folder'] = os.path.join(params['output_folder'], params['log_folder'])
params['model_folder'] = os.path.join(params['output_folder'], params['model_folder'])
if not os.path.exists(params['log_folder']):
    os.makedirs(params['log_folder'])
if not os.path.exists(params['model_folder']):
    os.makedirs(params['model_folder'])
painter = Painter(params) if params['visualise'] else None
grid_env = GridEnv(params, painter)

# model_path = os.path.join(params['output_folder'], "model", "Agent_dqn_state_dict_1600.mdl")
model_path = os.path.join("output_dqn", "model", "Agent_dqn_state_dict_123600.mdl")

dqn_agent = Agent(params, painter, model_path="")

writer = SummaryWriter(log_dir=params['log_folder'])

all_mean_rewards = []
all_mean_losses = []
time_step = 0
for i_episode in range(params['num_episodes']):
    observed_map, robot_pose = grid_env.reset()
    done = False
    rewards = []
    losses = []
    while not done:
        action = dqn_agent.act(observed_map, robot_pose)
        observed_map_next, robot_pose_next, reward, done = grid_env.step(action)
        dqn_agent.step(state=[observed_map, robot_pose], action=action, reward=reward,
                       next_state=[observed_map_next, robot_pose_next], done=done)
        # 转到下一个状态
        observed_map = observed_map_next.copy()
        robot_pose = robot_pose_next.copy()

        loss = dqn_agent.learn(memory_config_dir=params['memory_config_dir'])
        losses.append(loss)
        time_step += 1
        writer.add_scalar('train/loss_per_time_step', loss, time_step)

        # print("action=", action, ";reward:", reward, ";done:", done)
        if params['visualise']:
            painter.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
        rewards.append(reward)
        if done:
            if (i_episode + 1) % 200 == 0:
                # plt.cla()
                model_save_path = os.path.join(params['model_folder'], "Agent_dqn_state_dict_%d.mdl" % (i_episode + 1))
                dqn_agent.store_model(model_save_path)

            all_mean_rewards.append(np.mean(rewards))
            all_mean_losses.append(np.mean(losses))
            print()
            print(
                "i episode:{}; mean reward:{}; num_found_free_cell:{}/{};num_found_targets:{}/{}; num_found_occupied:{}/{}"
                    .format(i_episode, np.mean(rewards), grid_env.count_found_free, grid_env.count_free,
                            grid_env.count_found_target, grid_env.count_target,
                            grid_env.count_found_occupied, grid_env.count_occupied))
            writer.add_scalar('train/losses_smoothed', np.mean(all_mean_losses[max(0, i_episode - 200):]), i_episode)
            writer.add_scalar('train/loss_per_episode', np.mean(losses), i_episode)
            writer.add_scalar('train/rewards_smoothed', np.mean(all_mean_rewards[max(0, i_episode - 200):]), i_episode)
            writer.add_scalar("train/reward_per_episode", np.mean(rewards), i_episode)
            writer.add_scalar('train/num_found_free_cell', grid_env.count_found_free, i_episode)
            writer.add_scalar('train/num_found_targets', grid_env.count_found_target, i_episode)
            writer.add_scalar('train/num_found_total_cell', grid_env.count_found_free + grid_env.count_found_target,
                              i_episode)
            print("rewards:",rewards)

print('Complete')
