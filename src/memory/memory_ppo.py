import random

import torch
import numpy as np


class MemoryPPO:
    def __init__(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.data = self.init_data()

    def init_data(self):
        keys = ["frames", "robot_poses", "actions", "rewards", "frames_prime", "robot_poses_prime", "probs",
                "returns", "advantages"]
        data = {}
        for key in keys:
            data[key] = []
        return data

    def reset_data(self):
        self.data = self.init_data()

    def put_data(self, transitions):
        # if t_th == 0:
        # 只有t_th = 0 的时候才取得transition中h_in,c_in,h_out,c_out作为这个seq 的 t0, c0
        for i, key in enumerate(self.data.keys()):
            if key == "done":
                done_mask = 1 if transitions[i] else 0
                self.data[key].append([done_mask])
            else:
                self.data[key].append(transitions[i])

    def make_batch(self, train_device):
        frame_batch = self.data["frames"]
        pose_batch = self.data["robot_poses"]
        a_batch = self.data["actions"]
        r_batch = self.data["rewards"]
        frame_prime_batch = self.data["frames_prime"]
        pos_prime_batch = self.data["robot_poses_prime"]
        probs_batch = self.data["probs"]
        returns_batch = self.data["returns"]
        advantages_batch = self.data["advantages"]

        frame_batch = torch.tensor(np.array(frame_batch), dtype=torch.float).to(train_device)
        pose_batch = torch.tensor(np.array(pose_batch), dtype=torch.float).to(train_device)
        a_batch = torch.tensor(np.array(a_batch)).to(train_device)
        r_batch = torch.tensor(np.array(r_batch)).to(train_device)
        frame_prime_batch = torch.tensor(np.array(frame_prime_batch), dtype=torch.float).to(train_device)
        pos_prime_batch = torch.tensor(np.array(pos_prime_batch), dtype=torch.float).to(train_device)
        probs_batch = torch.tensor(np.array(probs_batch), dtype=torch.float).to(train_device)
        returns_batch = torch.tensor(np.array(returns_batch), dtype=torch.float).to(train_device)
        advantages_batch = torch.tensor(np.array(advantages_batch), dtype=torch.float).to(train_device)

        frame_batch = frame_batch.unsqueeze(2)
        pose_batch = pose_batch.unsqueeze(2)
        a_batch = a_batch.unsqueeze(2)
        r_batch = r_batch.unsqueeze(2)
        frame_prime_batch = frame_prime_batch.unsqueeze(2)
        pos_prime_batch = pos_prime_batch.unsqueeze(2)
        probs_batch = probs_batch.unsqueeze(2)
        returns_batch = returns_batch.unsqueeze(2)
        advantages_batch = advantages_batch.unsqueeze(2)

        frame_batch = frame_batch.permute(1, 0, 2, 3, 4)
        pose_batch = pose_batch.permute(1, 0, 2, 3)
        a_batch = a_batch.permute(1, 0, 2)
        r_batch = r_batch.permute(1, 0, 2)
        frame_prime_batch = frame_prime_batch.permute(1, 0, 2, 3, 4)
        pos_prime_batch = pos_prime_batch.permute(1, 0, 2, 3)
        probs_batch = probs_batch.permute(1, 0, 2, 3)
        returns_batch = returns_batch.permute(1, 0, 2)
        advantages_batch = advantages_batch.permute(1, 0, 2)

        return frame_batch, pose_batch, a_batch, r_batch, frame_prime_batch, pos_prime_batch, probs_batch, returns_batch, advantages_batch

    def __len__(self):
        return len(self.data["frames"])

    def is_full_batch(self):
        return len(self.data["frames"]) >= self.batch_size
