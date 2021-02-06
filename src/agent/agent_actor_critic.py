import random

import torch
import torch.optim as optim

import torch.nn.functional as F

import numpy as np


class ActorNetwork(torch.nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(1600, 784)
        self.fc2 = torch.nn.Linear(784, 256)
        self.fc3 = torch.nn.Linear(256, 32)

        self.fc_pose = torch.nn.Linear(2, 32)

        self.fc_pol = torch.nn.Linear(64, 4)

        # Initialize neural network weights
        # self.init_weights()

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
        out = self.fc_pol(out)
        return F.softmax(out, dim=1)


class ValueNetwork(torch.nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(1600, 784)
        self.fc2 = torch.nn.Linear(784, 256)
        self.fc3 = torch.nn.Linear(256, 32)

        self.fc_pose = torch.nn.Linear(2, 32)

        self.fc_value = torch.nn.Linear(64, 1)

        # Initialize neural network weights
        # self.init_weights()

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
        out = self.fc_value(out)
        return out


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
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.params['use_gpu'] else "cpu")
        print("device:", self.device)
        # 目标target
        self.actor_network = ActorNetwork().to(self.device)
        self.critic_network = ValueNetwork().to(self.device)
        if not model_path == "":
            self.load_model(file_path=model_path, map_location=self.device)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=1e-4)

        self.q_values = np.zeros((self.grid_h, self.grid_w, self.action_size))
        self.time_step = 0

    def act(self, frame, robot_pose):
        """
        :param frame: [w,h]
        :param robot_pose: [1,2]
        :return:
        """
        frame_in = torch.Tensor([frame]).to(self.device)
        robot_pose_in = torch.Tensor([robot_pose]).to(self.device)

        action_probs = self.actor_network(frame_in, robot_pose_in)

        categorical = torch.distributions.Categorical(action_probs)
        action = categorical.sample()

        return action.item(), categorical.log_prob(action)

    def step(self, state, action, log_prob, reward, next_state, done):

        # 计算Q value
        self.critic_network.train()
        self.actor_network.train()
        observed_map, robot_pose = state
        observed_map = torch.FloatTensor([observed_map]).to(self.device)
        robot_pose = torch.FloatTensor([robot_pose]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        observed_map_next, robot_pose_next = next_state
        observed_map_next = torch.FloatTensor([observed_map_next]).to(self.device)
        robot_pose_next = torch.FloatTensor([robot_pose_next]).to(self.device)

        current_value = self.critic_network(observed_map, robot_pose).to(self.device)
        next_value = self.critic_network(observed_map_next, robot_pose_next).to(self.device)

        log_prob = log_prob.to(self.device)
        # 这里要减去这一次的Q_value
        # print()
        # print("current_value:", current_value)
        # print("next_value:", next_value)
        #
        # print("current_value.requires_grad:", current_value.requires_grad)
        self.critic_optimizer.zero_grad()
        # print(reward)
        # print(next_value)
        # print(current_value)
        # td_error = torch.Tensor(reward + self.gamma * next_value - current_value).to(self.device)

        # 更新critic网络参数
        critic_loss = torch.sum(torch.square(reward + self.gamma * next_value - current_value))
        critic_loss.backward(retain_graph=True)

        # 更新actor网络参数
        self.actor_optimizer.zero_grad()
        actor_loss = torch.Tensor(log_prob * (reward + self.gamma * next_value - current_value))
        actor_loss.backward()
        # for name, parms in self.actor_network.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
        # print("actor loss:{}".format(actor_loss.item()))
        # print("critic loss:{}".format(critic_loss.item()))

        self.critic_optimizer.step()
        self.actor_optimizer.step()

        self.time_step += 1
        return actor_loss.item(), critic_loss.item()

    def load_model(self, file_path, map_location):
        state_dict = torch.load(file_path, map_location=map_location)
        self.policy_gradient_net.load_state_dict(state_dict)

    def store_model(self, actor_model_save_path, critic_model_save_path):
        torch.save(self.actor_network.state_dict(), actor_model_save_path)
        torch.save(self.critic_network.state_dict(), critic_model_save_path)

    def reset(self):
        pass

    def learn(self):
        self.actor_network.train()
        self.critic_network.train()
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
