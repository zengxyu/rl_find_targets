import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from torch.utils.tensorboard import SummaryWriter

from src.env.grid_env import GridEnv
from src.agent import Agent
import pygame
from src.env.config import *
from src.utils.pygame_painter import Painter

params = {
    'name': 'actor_critic',
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
    'batch_size': 128,
    'action_size': len(ACTION),

    'is_double': False,
    'is_priority_buffer': True,

    # grid params
    'max_step': 200,

    # train params
    'use_gpu': False,
    'visualise': False,
    'is_normalize': True,
    'num_episodes': 5000000,
    'scale': 15,

    # folder params

    # output
    'output_folder': "output_actor_critic",
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

model_path = os.path.join(params['output_folder'], "model", "Agent_dqn_state_dict_1600.mdl")
agent_ac = Agent(params, painter)

writer = SummaryWriter(log_dir=params['log_folder'])

all_mean_rewards = []
all_mean_actor_loss = []
all_mean_critic_loss = []
time_step = 0
for i_episode in range(params['num_episodes']):
    observed_map, robot_pose = grid_env.reset()
    done = False
    rewards = []
    actor_losses = []
    critic_losses = []
    while not done:
        action, action_log_prob = agent_ac.act(observed_map, robot_pose)
        observed_map_next, robot_pose_next, reward, done = grid_env.step(action)
        actor_loss, critic_loss = agent_ac.step(state=[observed_map, robot_pose], log_prob=action_log_prob,
                                                action=action, reward=reward,
                                                next_state=[observed_map_next, robot_pose_next], done=done)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        # 转到下一个状态
        observed_map = observed_map_next.copy()
        robot_pose = robot_pose_next.copy()

        time_step += 1

        if params['visualise']:
            painter.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
        rewards.append(reward)
        if done:
            all_mean_actor_loss.append(np.mean(actor_losses))
            all_mean_critic_loss.append(np.mean(critic_losses))
            writer.add_scalar('train/actor_loss_per_episode', np.mean(actor_losses), time_step)
            writer.add_scalar('train/critic_loss_per_episode', np.mean(critic_loss), time_step)
            if (i_episode + 1) % 200 == 0:
                # plt.cla()
                actor_model_save_path = os.path.join(params['model_folder'],
                                                     "Agent_actor_state_dict_%d.mdl" % (i_episode + 1))
                critic_model_save_path = os.path.join(params['model_folder'],
                                                      "Agent_critic_state_dict_%d.mdl" % (i_episode + 1))
                agent_ac.store_model(actor_model_save_path, critic_model_save_path)

            all_mean_rewards.append(np.mean(rewards))
            print()
            print(
                "i episode:{}; mean reward:{}; num_found_free_cell:{}/{};num_found_targets:{}/{}; num_found_occupied:{}/{}"
                    .format(i_episode, np.mean(rewards), grid_env.count_found_free, grid_env.count_free,
                            grid_env.count_found_target, grid_env.count_target,
                            grid_env.count_found_occupied, grid_env.count_occupied))
            writer.add_scalar('train/losses_actor_smoothed', np.mean(all_mean_actor_loss[max(0, i_episode - 200):]),
                              i_episode)
            writer.add_scalar('train/losses_critic_smoothed', np.mean(all_mean_critic_loss[max(0, i_episode - 200):]),
                              i_episode)
            writer.add_scalar('train/rewards_smoothed', np.mean(all_mean_rewards[max(0, i_episode - 200):]), i_episode)
            writer.add_scalar("train/reward_per_episode", np.mean(rewards), i_episode)
            writer.add_scalar('train/num_found_free_cell', grid_env.count_found_free, i_episode)
            writer.add_scalar('train/num_found_targets', grid_env.count_found_target, i_episode)
            writer.add_scalar('train/num_found_total_cell', grid_env.count_found_free + grid_env.count_found_target,
                              i_episode)

print('Complete')
