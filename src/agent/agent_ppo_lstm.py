import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn.functional as F
from src.env.config import ACTION
from src.network.network_ppo_lstm import PPO_LSTM
from src.memory.memory_ppo_lstm import MemoryPPOLSTM


class Agent:
    """description of class"""

    def __init__(self, params, writer, train_agent=False, is_resume=False, filepath="",
                 train_device=torch.device('cuda')):
        self.name = "PPOng"
        self.train_agent = train_agent
        self.writer = writer
        # self.frame_size = np.product(field.shape)
        self.pose_size = 2
        self.action_size = len(ACTION)
        self.TRAJ_COLLECTION_NUM = 16
        self.TRAJ_LEN = 4
        self.batch_size = params['batch_size']
        self.seq_len = params['seq_len']

        # self.BATCH_SIZE = 32
        self.tlen_counter = 0
        self.tcol_counter = 0
        # self.FRAME_SKIP = 1
        self.EPSILON = 0.2
        self.EPOCH_NUM = 4
        self.K_epoch = 4
        self.gamma = 0.98
        self.train_device = train_device  # 'cuda' if torch.cuda.is_available() else 'cpu'
        print("train device : {}".format(self.train_device))
        self.policy = self.get_model(is_resume, filepath)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.memory = MemoryPPOLSTM(batch_size=self.batch_size, seq_len=self.seq_len)

        self.frames = []
        self.robot_poses = []
        self.actions = []
        self.values = []
        self.probs = []
        self.deltas = []
        self.last_frame = None
        self.last_robot_pose = None
        self.last_action = None
        self.last_probs = None
        self.last_reward = None
        self.last_value = None

    def get_model(self, is_resume, filepath):
        policy = PPO_LSTM(self.action_size).to(self.train_device)
        if is_resume:
            self.load_model(filename=filepath, map_location=self.train_device)
        return policy

    def load_model(self, filename, map_location=None):  # pass 'cpu' as map location if no cuda available
        state_dict = torch.load(filename, map_location)
        self.policy.load_state_dict(state_dict)

    def store_model(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def reset(self):
        self.last_frame = None
        self.last_robot_pose = None
        self.last_action = None
        self.last_probs = None
        self.last_reward = None
        self.last_value = None

    def store_data(self, transition):
        self.memory.put_data(transition)

    def train_net(self):
        # print("train net")
        loss_v = 0
        for i in range(self.batch_size):
            frame, pos, a, r, frame_prime, pos_prime, prob_a, done_mask, h_in, c_in, h_out, c_out = self.memory.make_batch(
                i * self.seq_len, self.train_device)

            value_prime, _, _, _ = self.policy(frame_prime, pos_prime, h_out, c_out)
            td_target = r + self.gamma * value_prime * done_mask
            value, pol, _, _ = self.policy(frame, pos, h_in, c_in)

            delta = td_target - value
            delta = delta.detach().cpu().numpy()

            advantage = 0.0

            advantage_lst = []
            for item in delta[::-1]:
                advantage = self.gamma * advantage + item[0]
                advantage_lst.append([advantage])

            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            a = a.squeeze()
            advantage = advantage.squeeze()
            pol = pol.squeeze()
            prob_a = prob_a.squeeze()

            new_categorical = torch.distributions.Categorical(pol)
            prob_new = new_categorical.log_prob(a)

            ratio = torch.exp(prob_new - prob_a)  # a/b == log(exp(a)-exp(b))
            advantage = advantage.to(self.train_device)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.EPSILON, 1 + self.EPSILON) * advantage
            loss_ent = -new_categorical.entropy().mean()
            loss = -torch.min(surr1, surr2) + F.mse_loss(value, td_target.detach()) + 0.01 * loss_ent

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)

            self.optimizer.step()

            # print("loss mean:{}".format(loss.mean().item()))
            loss_v += loss.mean().item()

        return loss_v / self.K_epoch

    def act(self, frame, robot_pose, h_in, c_in):
        # frame : [seq_len, batch_size, dim, h, w]
        frame_in = torch.Tensor([[[frame]]]).to(self.train_device)
        robot_pose_in = torch.Tensor([[robot_pose]]).to(self.train_device)
        h_in, c_in = h_in.to(self.train_device), c_in.to(self.train_device)

        value, pol, h_out, c_out = self.policy(frame_in, robot_pose_in, h_in, c_in)

        categorical = torch.distributions.Categorical(pol)
        action = categorical.sample()
        log_prob = categorical.log_prob(action)
        return action, log_prob, h_out, c_out

    def update_policy(self):
        for i in range(self.EPOCH_NUM):
            mb_frames, mb_robot_poses, mb_actions, mb_returns, mb_probs, mb_advantages = self.traj_memory

            new_vals, new_probs = self.policy(torch.cat(mb_frames, dim=0).to(self.train_device),
                                              torch.cat(mb_robot_poses, dim=0).to(self.train_device))
            old_probs = torch.cat(mb_probs, dim=0)

            new_pol = torch.distributions.Categorical(new_probs)
            old_pol = torch.distributions.Categorical(old_probs)

            action_tensor = torch.cat(mb_actions, dim=0)

            ratio = torch.exp(new_pol.log_prob(action_tensor) - old_pol.log_prob(action_tensor))

            advantage_tensor = torch.cat(mb_advantages, dim=0)

            loss_clip = -torch.min(ratio * advantage_tensor,
                                   torch.clamp(ratio, 1 - self.EPSILON, 1 + self.EPSILON) * advantage_tensor).mean()

            returns_tensor = torch.cat(mb_returns, dim=0)

            loss_val = F.mse_loss(new_vals.squeeze(), returns_tensor)

            loss_ent = -new_pol.entropy().mean()

            c1, c2 = 1, 0.01

            loss = loss_clip + c1 * loss_val + c2 * loss_ent

            # loss.backward(retain_graph=True)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.traj_memory = [[], [], [], [], [], []]
        self.tcol_counter = 0

    def get_name(self):
        return self.name
