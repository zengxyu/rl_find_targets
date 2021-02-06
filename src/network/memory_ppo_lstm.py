import random

import torch


class MemoryPPOLSTM:
    def __init__(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.data = self.init_data()

    def init_data(self):
        keys = ["frame", "pos", "a", "r", "frame_prime", "pos_prime", "log_prob", "done",
                "h_in", "c_in", "h_out", "c_out"]
        data = {}
        for key in keys:
            data[key] = []
        return data

    def reset_data(self):
        self.data = self.init_data()

    def put_data(self, transition):
        # if t_th == 0:
        # 只有t_th = 0 的时候才取得transition中h_in,c_in,h_out,c_out作为这个seq 的 t0, c0
        for i, key in enumerate(self.data.keys()):
            if key == "done":
                done_mask = 0 if transition[i] else 1
                self.data[key].append([done_mask])
            else:
                self.data[key].append(transition[i])

    def make_batch(self, i, train_device):
        frame_batch, pose_batch, a_batch, r_batch, frame_prime_batch, pos_prime_batch, done_mask_batch, prob_a_batch = [], [], [], [], [], [], [], []
        h_in_batch, c_in_batch, h_out_batch, c_out_batch = [], [], [], []
        # for bs in range(self.batch_size):
        seq_start = i
        seq_end = seq_start + self.seq_len
        frame_batch.append(self.data["frame"][seq_start:seq_end])
        pose_batch.append(self.data["pos"][seq_start:seq_end])
        a_batch.append(self.data["a"][seq_start:seq_end])
        r_batch.append(self.data["r"][seq_start:seq_end])
        frame_prime_batch.append(self.data["frame_prime"][seq_start:seq_end])
        pos_prime_batch.append(self.data["pos_prime"][seq_start:seq_end])
        done_mask_batch.append(self.data["done"][seq_start:seq_end])
        prob_a_batch.append(self.data["log_prob"][seq_start:seq_end])
        h_in_batch.append(self.data["h_in"][seq_start])
        c_in_batch.append(self.data["c_in"][seq_start])
        h_out_batch.append(self.data["h_out"][seq_start])
        c_out_batch.append(self.data["c_out"][seq_start])

        frame_batch = torch.tensor(frame_batch, dtype=torch.float).to(train_device)
        pose_batch = torch.tensor(pose_batch, dtype=torch.float).to(train_device)
        a_batch = torch.tensor(a_batch).to(train_device)
        r_batch = torch.tensor(r_batch).to(train_device)
        frame_prime_batch = torch.tensor(frame_prime_batch, dtype=torch.float).to(train_device)
        pos_prime_batch = torch.tensor(pos_prime_batch, dtype=torch.float).to(train_device)
        done_mask_batch = torch.tensor(done_mask_batch, dtype=torch.bool).to(train_device)
        prob_a_batch = torch.tensor(prob_a_batch).to(train_device)

        h_in_batch = torch.Tensor(h_in_batch).to(train_device)
        c_in_batch = torch.Tensor(c_in_batch).to(train_device)
        h_out_batch = torch.Tensor(h_out_batch).to(train_device)
        c_out_batch = torch.Tensor(c_out_batch).to(train_device)

        frame_batch = frame_batch.permute(1, 0, 2, 3, 4)
        pose_batch = pose_batch.permute(1, 0, 2, 3)
        a_batch = a_batch.permute(1, 0, 2)
        r_batch = r_batch.permute(1, 0, 2)
        frame_prime_batch = frame_prime_batch.permute(1, 0, 2, 3, 4)
        pos_prime_batch = pos_prime_batch.permute(1, 0, 2, 3)
        done_mask_batch = done_mask_batch.permute(1, 0, 2)
        prob_a_batch = prob_a_batch.permute(1, 0, 2)
        h_in_batch = h_in_batch.permute(1, 0, 2)
        c_in_batch = c_in_batch.permute(1, 0, 2)
        h_out_batch = h_out_batch.permute(1, 0, 2)
        c_out_batch = c_out_batch.permute(1, 0, 2)

        return frame_batch, pose_batch, a_batch, r_batch, frame_prime_batch, pos_prime_batch, done_mask_batch, prob_a_batch, \
               h_in_batch, c_in_batch, h_out_batch, c_out_batch

    def __len__(self):
        return len(self.data["frame"])

    def is_full_seq(self):
        return len(self.data["frame"]) >= self.seq_len

    def is_full_batch(self):
        return len(self.data["frame"]) >= self.batch_size * self.seq_len
