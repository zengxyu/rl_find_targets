import random

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from src.memory.normalizer import Normalizer


class Network_Policy_Gradient(torch.nn.Module):
    def __init__(self):
        super(Network_Policy_Gradient, self).__init__()
        self.fc1 = torch.nn.Linear(1600, 784)
        self.fc2 = torch.nn.Linear(784, 256)
        self.fc3 = torch.nn.Linear(256, 32)

        self.fc_pose = torch.nn.Linear(2, 32)

        self.fc_pol = torch.nn.Linear(64, 4)

        # Initialize neural network weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, frame, robot_pose):
        frame = frame.reshape(frame.size(0), -1)
        robot_pose = robot_pose.reshape(robot_pose.size(0), -1)

        out = self.fc1(frame)
        out = F.relu(out)

        out = self.fc2(out)
        out = F.relu(out)

        out = self.fc3(out)
        out = F.relu(out)

        out_pose = self.fc_pose(robot_pose)
        out_pose = F.relu(out_pose)
        out = torch.cat((out, out_pose), dim=1)
        # ===================================================changed
        action_scores = self.fc_pol(out)
        return F.softmax(action_scores, dim=1)


class EpisodeMemory:
    def __init__(self):
        self.saved_log_probs = []
        self.rewards = []

    def clear(self):
        self.saved_log_probs = []
        self.rewards = []


class Agent:
    def __init__(self, params, painter, model_path=""):
        self.name = "grid world"
        self.params = params
        self.painter = painter
        self.grid_w = params['w']
        self.grid_h = params['h']
        self.update_every = params['update_every']
        self.eps = params['eps_start']
        self.eps_decay = params['eps_decay']
        self.eps_min = params['eps_min']
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.seed = random.seed(42)

        self.is_normalize = params['is_normalize']

        self.action_size = params['action_size']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", self.device)
        # 目标target
        self.policy_gradient_net = Network_Policy_Gradient().to(self.device)
        if not model_path == "":
            self.load_model(file_path=model_path, map_location=self.device)

        self.normalizer = Normalizer(config_dir=params['memory_config_dir']) if self.is_normalize else None

        self.optimizer = optim.Adam(self.policy_gradient_net.parameters(), lr=1e-4)
        # if params['is_priority_buffer']:
        self.episode_memory = EpisodeMemory()

        self.q_values = np.zeros((self.grid_h, self.grid_w, self.action_size))
        self.time_step = 0

    def step(self, state, action, reward, next_state, done):
        self.episode_memory.rewards.append(reward)
        self.time_step += 1

    def load_model(self, file_path, map_location):
        state_dict = torch.load(file_path, map_location=map_location)
        self.policy_gradient_net.load_state_dict(state_dict)

    def store_model(self, file_path):
        torch.save(self.policy_gradient_net.state_dict(), file_path)

    def reset(self):
        pass

    def act(self, frame, robot_pose):
        """
        :param frame: [w,h]
        :param robot_pose: [1,2]
        :return:
        """
        frame_in = torch.Tensor([frame]).to(self.device)
        robot_pose_in = torch.Tensor([robot_pose]).to(self.device)

        if self.is_normalize:
            frame_in = self.normalizer.normalize_frame_in(frame_in)
            robot_pose_in = self.normalizer.normalize_robot_pose_in(robot_pose_in)

        self.policy_gradient_net.eval()
        with torch.no_grad():
            action_probs = self.policy_gradient_net(frame_in, robot_pose_in)

        categorical = torch.distributions.Categorical(action_probs)
        action = categorical.sample()
        self.episode_memory.saved_log_probs.append(categorical.log_prob(action))

        return action.item()


    def learn(self):
        self.policy_gradient_net.train()
        policy_loss = []
        R = 0
        returns = []
        for r in self.episode_memory.rewards[::-1]:
            R = r + self.gamma * R
            # 将R插入到指定的位置0处
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for log_prob, R in zip(self.episode_memory.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum().requires_grad_()
        policy_loss.backward()
        self.optimizer.step()

        self.episode_memory.clear()
        return policy_loss.item()
