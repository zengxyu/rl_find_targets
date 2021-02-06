import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from src.env.grid_env import GridEnv
from src.agent import Agent
import pygame
from src.env.config import *
from src.utils.pygame_painter import Painter
from tqdm import tqdm

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
    'eps_start': 1.0,  # Default/starting value of eps
    'eps_decay': 0.99999,  # Epsilon decay rate
    'eps_min': 1.0,  # Minimum epsilon
    'gamma': 0.98,
    'buffer_size': 500000,
    'batch_size': 128,
    'action_size': len(ACTION),

    'is_double': False,
    # grid params
    'max_step': 200,

    # train params
    'visualise': False,
    'num_episodes': 5000000,
    'scale': 15,
    'is_normalize': False,

    # folder params

    # output
    'output_folder': "output_dqn",
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
if not os.path.exists(params['memory_config_dir']):
    os.makedirs(params['memory_config_dir'])

painter = None
if params['visualise']:
    painter = Painter(params)

grid_env = GridEnv(params, painter)
dqn_agent = Agent(params, painter)

time_step = 0
t_bar = tqdm(range(int(params['buffer_size'] / params['max_step'])))
for i_episode in t_bar:
    t_bar.set_description("i_episode:{}".format(i_episode))
    observed_map, robot_pose = grid_env.reset()
    done = False
    while not done:
        action = dqn_agent.act(observed_map, robot_pose)
        observed_map_next, robot_pose_next, reward, done = grid_env.step(action)
        dqn_agent.step(state=[observed_map, robot_pose], action=action, reward=reward,
                       next_state=[observed_map_next, robot_pose_next], done=done)
        # 转到下一个状态
        observed_map = observed_map_next.copy()
        robot_pose = robot_pose_next.copy()

        if params['visualise']:
            painter.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
    if (i_episode + 1) % 500 == 0:
        dqn_agent.save_normalized_memory(save_dir=params['memory_config_dir'])

print('Complete')
