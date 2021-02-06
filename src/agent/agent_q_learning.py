import random

from src.env.config import *


class Agent:
    def __init__(self, env_h, env_w, params, model_path=None):
        self.q_func = dict()
        self.env_h = env_h
        self.env_w = env_w
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        if model_path is None:
            self.init_q_func()
        else:
            self.load_model(model_path)

    def init_q_func(self):
        # 初始化行为值函数为0
        for i in range(self.env_h):
            for j in range(self.env_w):
                for a in range(len(ACTION)):
                    key = "{}_{}_{}".format(i, j, a)
                    self.q_func[key] = 0.0

    def act_greedy(self, robot_pose):
        a_max = ACTION.MOVE_UP
        key = "{}_{}_{}".format(robot_pose[0], robot_pose[1], a_max)
        q_max = self.q_func[key]

        for a in range(len(ACTION)):
            key = "{}_{}_{}".format(robot_pose[0], robot_pose[1], a)
            q = self.q_func[key]
            if q_max < q:
                q_max = q
                a_max = a
        return a_max

    def act_eps_greedy(self, robot_pose, epi):
        a_max = ACTION.MOVE_UP
        key = "{}_{}_{}".format(robot_pose[0], robot_pose[1], a_max)
        q_max = self.q_func[key]

        for a in range(len(ACTION)):
            key = "{}_{}_{}".format(robot_pose[0], robot_pose[1], a)
            q = self.q_func[key]
            if q_max < q:
                q_max = q
                a_max = a
        # 概率部分
        pro = dict()
        for a in range(len(ACTION)):
            pro[a] = 0.0

        pro[a_max] += 1 - epi
        for a in range(len(ACTION)):
            pro[a] += epi / len(ACTION)

        # 选择动作
        r = random.random()
        s = 0.0
        for a in range(len(ACTION)):
            s += pro[a]
            if s >= r:
                return a

        return ACTION.MOVE_UP

    def learn(self, key1, key2, reward):

        # 利用qlearning方法更新值函数
        self.q_func[key1] = self.q_func[key1] + self.alpha * (
                reward + self.gamma * self.q_func[key2] - self.q_func[key1])

    def store_model(self, model_path):
        pickle.dump(self.q_func, open(model_path, 'wb'))

    def load_model(self, model_path):
        self.q_func = pickle.load(open(model_path, 'rb'))
