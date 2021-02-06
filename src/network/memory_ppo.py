import torch


class MemoryPPO:
    def __init__(self, traj_collection_num, traj_len):
        self.frames = []
        self.robot_poses = []
        self.actions = []
        self.values = []
        self.probs = []
        self.deltas = []

        self.traj_memory = [[], [], [], [], [], []]

        self.traj_collection_num = traj_collection_num
        self.traj_len = traj_len

        self.tlen_counter = 0
        self.tcol_counter = 0

    def store_trajectory_to_memory(self, last_frame, last_robot_pose, last_action, last_value, last_probs, last_reward,
                                   final_state):
        T = len(self.deltas)
        advantages = torch.zeros(T, 1).to(self.train_device)
        returns = torch.zeros(T, 1).to(self.train_device)
        advantages[T - 1] = self.deltas[T - 1]
        returns[T - 1] = advantages[T - 1] + self.values[T - 1]
        for i in range(1, T):
            advantages[T - i - 1] = self.deltas[T - i - 1] + self.gamma * advantages[T - i]
            returns[T - i - 1] = advantages[T - i - 1] + self.values[T - i - 1]

        self.traj_memory[0].append(last_frame)
        self.traj_memory[1].append(last_robot_pose)
        self.traj_memory[2].append(last_action)
        self.traj_memory[3].append(last_value)
        self.traj_memory[4].append(last_probs)
        self.traj_memory[5].append(last_value - last_value)

    def reset_memory(self):
        self.traj_memory = [[], [], [], [], [], []]
        self.tlen_counter = 0
        self.tcol_counter = 0

    def get_traj_memory(self):
        return self.traj_memory
