import os
import pickle
import numpy as np
import torch


class Normalizer:
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.frames_mean, self.frames_std = None, None
        self.robot_pos_mean, self.robot_pos_std = None, None
        self.actions_mean, self.actions_std = None, None
        self.rewards_mean, self.rewards_std = None, None
        self.frames_next_mean, self.frames_next_std = None, None
        self.robot_pos_next_mean, self.robot_pos_next_std = None, None
        self.load_normalization_config()

    def save_normalization_config(self, config_dir):
        frames = []
        robot_pos = []
        actions = []
        rewards = []
        frames_next = []
        robot_pos_next = []
        for experience in self.memory:
            if experience is not None:
                frames.append(experience.state[0])
                robot_pos.append(experience.state[1])
                actions.append(experience.action)
                rewards.append(experience.reward)
                frames_next.append(experience.next_state[0])
                robot_pos_next.append(experience.next_state[1])
        frames_mean = np.mean(frames)
        frames_std = np.std(frames)

        robot_pos_mean = np.mean(robot_pos)
        robot_pos_std = np.std(robot_pos)

        actions_mean = np.mean(actions)
        actions_std = np.std(actions)

        rewards_mean = np.mean(rewards)
        rewards_std = np.std(rewards)

        frames_next_mean = np.mean(frames_next)
        frames_next_std = np.std(frames_next)

        robot_pos_next_mean = np.mean(robot_pos_next)
        robot_pos_next_std = np.std(robot_pos_next)

        with open(os.path.join(config_dir, "frames_mean_std.pkl"), 'wb') as f:
            pickle.dump([frames_mean, frames_std], f)
        with open(os.path.join(config_dir, "robot_pos_mean_std.pkl"), 'wb') as f:
            pickle.dump([robot_pos_mean, robot_pos_std], f)
        with open(os.path.join(config_dir, "actions_mean_std.pkl"), 'wb') as f:
            pickle.dump([actions_mean, actions_std], f)
        with open(os.path.join(config_dir, "rewards_mean_std.pkl"), 'wb') as f:
            pickle.dump([rewards_mean, rewards_std], f)
        with open(os.path.join(config_dir, "frames_next_mean_std.pkl"), 'wb') as f:
            pickle.dump([frames_next_mean, frames_next_std], f)
        with open(os.path.join(config_dir, "robot_pos_next_mean_std.pkl"), 'wb') as f:
            pickle.dump([robot_pos_next_mean, robot_pos_next_std], f)

    def load_normalization_config(self):
        print(self.config_dir)
        with open(os.path.join(self.config_dir, "frames_mean_std.pkl"), 'rb') as f:
            self.frames_mean, self.frames_std = pickle.load(f)
        with open(os.path.join(self.config_dir, "robot_pos_mean_std.pkl"), 'rb') as f:
            self.robot_pos_mean, self.robot_pos_std = pickle.load(f)
        with open(os.path.join(self.config_dir, "actions_mean_std.pkl"), 'rb') as f:
            self.actions_mean, self.actions_std = pickle.load(f)
        with open(os.path.join(self.config_dir, "rewards_mean_std.pkl"), 'rb') as f:
            self.rewards_mean, self.rewards_std = pickle.load(f)
        with open(os.path.join(self.config_dir, "frames_next_mean_std.pkl"), 'rb') as f:
            self.frames_next_mean, self.frames_next_std = pickle.load(f)
        with open(os.path.join(self.config_dir, "robot_pos_next_mean_std.pkl"), 'rb') as f:
            self.robot_pos_next_mean, self.robot_pos_next_std = pickle.load(f)

    def normalize_frame_in(self, frame_in):
        return (frame_in - self.frames_mean) / self.frames_std

    def normalize_robot_pose_in(self, robot_pose_in):
        return (robot_pose_in - self.robot_pos_mean) / self.robot_pos_std

    def normalize_reward(self, reward):
        return (reward - self.rewards_mean) / self.rewards_std

    def normalize_frame_next(self, frame_next):
        return (frame_next - self.frames_next_mean) / self.frames_next_std

    def normalize_robot_pose_next(self, robot_poses_next):
        return (robot_poses_next - self.robot_pos_next_mean) / self.robot_pos_next_std

    def normalize_mini_batch(self, mini_batch):
        frames_in, robot_poses_in, actions, rewards, next_frames_in, next_robot_poses_in, dones = mini_batch
        frames_in = self.normalize_frame_in(frames_in)
        robot_poses_in = self.normalize_robot_pose_in(robot_poses_in)
        rewards = self.normalize_reward(rewards)
        next_frames_in = self.normalize_frame_next(next_frames_in)
        next_robot_poses_in = self.normalize_robot_pose_in(next_robot_poses_in)
        return [frames_in, robot_poses_in, actions, rewards, next_frames_in, next_robot_poses_in, dones]
