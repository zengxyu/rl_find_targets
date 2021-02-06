import logging
from src.env.config import *
import numpy as np

logger = logging.getLogger(__name__)


class GridEnv:

    def __init__(self, params, painter):
        self.h = params['h']
        self.w = params['w']
        self.title = params['name']
        # state space
        self.map = np.array(params['field_data'])
        self.observed_map = np.ones((self.h, self.w)) * STATEVALUE.UNKNOWN
        self.transition = self.init_transition_func()
        self.max_step = params['max_step']

        self.visualise = params['visualise']
        # 当前状态，首次初始化状态
        self.pos = params['start_pos']

        # 当前方向，首次初始化方向
        self.direction = params['start_direction']
        # 当前计步
        self.count_step = 0

        # robot and robot camera
        self.num_all_cell = self.h * self.w
        self.count_free, self.count_target, self.count_occupied = self.get_all_free_and_target_cell_nums()
        self.count_found_free = 0
        self.count_found_target = 0
        self.count_found_occupied = 0
        self.offset_left = -4
        self.offset_right = 4

        self.region_up, self.region_down, self.region_left, self.region_right = self.get_camera_view_region()

        if self.visualise:
            self.painter = painter

    def get_all_free_and_target_cell_nums(self):
        count_target = 0
        count_free = 0
        count_occupied = 0
        for i in range(self.h):
            for j in range(self.w):
                if self.map[i, j] == STATEVALUE.TARGET:
                    count_target += 1
                elif self.map[i, j] == STATEVALUE.FREE:
                    count_free += 1
                elif self.map[i, j] == STATEVALUE.OCCUPIED:
                    count_occupied += 1
        return count_free, count_target, count_occupied

    def reset(self):

        self.observed_map = np.ones((self.h, self.w)) * STATEVALUE.UNKNOWN

        # 当前状态，重新初始化状态
        self.pos = np.random.randint(0, min(self.h, self.w), size=[2, ])
        # 当前方向，重新初始化方向
        self.direction = MOVEMENTS[np.random.randint(0, len(ACTION))]

        # 当前计步
        self.count_step = 0

        # robot and robot camera
        self.count_found_occupied = 0
        self.count_found_target = 0
        self.count_found_free = 0

        if self.visualise:
            self.painter.draw_field(self.pos, self.direction)
            self.painter.draw_observed_map(self.pos, self.direction)
            self.painter.set_top_text("{}/{}".format(self.count_step, self.max_step),
                                      "{}/{}".format(self.count_found_target, self.count_target),
                                      "{}/{}".format(self.count_found_free, self.count_free),
                                      "{}/{}".format(self.count_found_occupied, self.count_occupied),
                                      self.title)

        return self.observed_map, self.pos

    def step(self, cur_action):
        # print("cur_action:{}".format(cur_action))
        # current state
        pos = self.pos

        # 获得下一个状态 && # 更新当前状态为下一个状态
        self.pos = self.get_next_pos(cur_pos=pos, cur_action=cur_action)

        self.count_step += 1

        # 判断是否为终止状态
        is_terminal = self.count_step > self.max_step or self.count_found_target == self.count_target

        # 更新机器人状态图
        new_observed_map = self.update_observed_map(cur_pos=self.pos, face_towards=cur_action)

        # 获得奖励
        r = self.get_reward(self.observed_map)
        # print("reward ================================================ {}".format(r))

        if self.visualise:
            self.painter.draw_field(self.pos, direction=MOVEMENTS[cur_action])
            self.painter.draw_observed_map(self.pos, MOVEMENTS[cur_action], self.observed_map, new_observed_map)
            self.painter.set_top_text("{}/{}".format(self.count_step, self.max_step),
                                      "{}/{}".format(self.count_found_target, self.count_target),
                                      "{}/{}".format(self.count_found_free, self.count_free),
                                      "{}/{}".format(self.count_found_occupied, self.count_occupied),
                                      self.title)

            # self.painter.update()
        observed_map = self.observed_map.copy()
        observed_map[self.pos[0], self.pos[1]] = STATEVALUE.ROBOT
        #  observed_map_next, robot_pose_next, reward, done
        return observed_map, self.pos, r, is_terminal

    def get_reward(self, observed_map):
        # iterate the states the camera can see, and check if there any states with target
        count_target = 0
        count_free = 0
        count_occupied = 0
        for i in range(observed_map.shape[0]):
            for j in range(observed_map.shape[1]):
                if observed_map[i, j] == STATEVALUE.TARGET:
                    count_target += 1
                if observed_map[i, j] == STATEVALUE.FREE:
                    count_free += 1
                if observed_map[i, j] == STATEVALUE.OCCUPIED:
                    count_occupied += 1
        count_free_diff = count_free - self.count_found_free
        count_target_diff = count_target - self.count_found_target
        count_occupied_diff = count_occupied - self.count_found_occupied
        # reward = 1 / (count_target_diff * REWARD.TARGET + count_free_diff * REWARD.FREE + 1e-6)
        reward = count_target_diff * REWARD.TARGET + count_free_diff * REWARD.FREE + count_occupied_diff * REWARD.OCCUPIED
        # reward = np.e ** reward
        self.count_found_target, self.count_found_free, self.count_found_occupied = count_target, count_free, count_occupied
        return reward

    # def get_reward(self, observed_map, robot_pos):
    #     square_dist = (robot_pos[0] - int(self.h / 2)) ** 2 + (robot_pos[1] - int(self.w / 2)) ** 2 + 1
    #     reward = 1 / square_dist
    #     return reward

    # def get_next_pos(self, cur_pos, cur_action):

    #     next_pos = cur_pos + MOVEMENTS[cur_action]
    #     if next_pos[0] < 0 or next_pos[1] < 0 or next_pos[0] >= self.h or next_pos[1] >= self.w:
    #         next_pos = cur_pos
    #     elif self.map[next_pos[0], next_pos[1]] == STATEVALUE.TARGET:
    #         next_pos = cur_pos
    #
    #     return next_pos

    def get_next_pos(self, cur_pos, cur_action):
        key = "{}_{}_{}".format(cur_pos[0], cur_pos[1], cur_action)
        return np.array(self.transition[key])

    def init_transition_func(self):
        """
        initialize transition function 初始化转换函数
        :return:
        """
        t = dict()
        for i in range(self.h):
            for j in range(self.w):
                for a in range(len(ACTION)):
                    key = "{}_{}_{}".format(i, j, a)
                    movement = MOVEMENTS[a]
                    next_pos = [i + movement[0], j + movement[1]]
                    if next_pos[0] < 0 or next_pos[1] < 0 or next_pos[0] >= self.h or next_pos[1] >= self.w:
                        next_pos = [i, j]
                    # elif self.map[next_pos[0], next_pos[1]] == STATEVALUE.TARGET:
                    #     next_pos = [i, j]
                    t[key] = next_pos
        return t

    def get_camera_view_region(self):
        """这个函数没有检查"""
        region_up = []
        region_down = []
        region_left = []
        region_right = []
        region_remain = []
        count = 0
        for m in np.arange(self.offset_left + 1, self.offset_right):
            for n in np.arange(self.offset_left + 1, self.offset_right):
                count += 1
                if self.offset_left + 1 <= n <= 0 and m - n > 0 and m + n <= 0:
                    region_up.append([n, m])
                elif 0 <= m <= self.offset_right and m + n > 0 and m - n >= 0:
                    region_right.append([n, m])
                elif 0 <= n <= self.offset_right and m - n < 0 and m + n >= 0:
                    region_down.append([n, m])
                elif self.offset_left + 1 <= m <= 0 and m + n < 0 and m - n <= 0:
                    region_left.append([n, m])
                else:
                    region_remain.append([n, m])
        return region_up, region_down, region_left, region_right

    def update_observed_map(self, cur_pos, face_towards):
        """这个函数没有检查"""
        new_observed_map = np.ones_like(self.observed_map) * STATEVALUE.UNKNOWN
        i, j = cur_pos
        if face_towards == ACTION.MOVE_UP:
            for n, m in self.region_up:
                if 0 <= i + n <= self.h - 1 and 0 <= j + m <= self.w - 1:
                    self.observed_map[i + n, j + m] = self.map[i + n, j + m]
                    new_observed_map[i + n, j + m] = self.map[i + n, j + m]

        elif face_towards == ACTION.MOVE_DOWN:
            for n, m in self.region_down:
                if 0 <= i + n <= self.h - 1 and 0 <= j + m <= self.w - 1:
                    self.observed_map[i + n, j + m] = self.map[i + n, j + m]
                    new_observed_map[i + n, j + m] = self.map[i + n, j + m]


        elif face_towards == ACTION.MOVE_LEFT:
            for n, m in self.region_left:
                if 0 <= i + n <= self.h - 1 and 0 <= j + m <= self.w - 1:
                    self.observed_map[i + n, j + m] = self.map[i + n, j + m]
                    new_observed_map[i + n, j + m] = self.map[i + n, j + m]

        else:
            for n, m in self.region_right:
                if 0 <= i + n <= self.h - 1 and 0 <= j + m <= self.w - 1:
                    self.observed_map[i + n, j + m] = self.map[i + n, j + m]
                    new_observed_map[i + n, j + m] = self.map[i + n, j + m]
        # self.observed_map[i, j] = STATEVALUE.ROBOT
        return new_observed_map
