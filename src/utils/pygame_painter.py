import pickle

import numpy as np
import pygame
from src.env.grid_env import STATEVALUE

LINE_COLOR = (150, 150, 150)
DIVIDE_LINE_COLOR = (255, 255, 255)
UNKNOWN_COLOR = 0x000000
FREE_COLOR = 0x696969
OCCUPIED_COLOR = (39, 228, 75)
TARGET_COLOR = (179, 88, 247)

ROBOT_COLOR = (255, 0, 0)
FONT_COLOR = (255, 0, 0)
NEW_OBSERVED_SCENE_COLOR = 0xF7ED58
COLOR_DICT = {STATEVALUE.UNKNOWN: UNKNOWN_COLOR, STATEVALUE.FREE: FREE_COLOR,
              STATEVALUE.OCCUPIED: OCCUPIED_COLOR, STATEVALUE.TARGET: TARGET_COLOR}


class Painter:
    def __init__(self, params):
        self.w = params['w']
        self.h = params['h']
        self.scale = params['scale']
        self.border_width = 2
        self.line_width = 1
        self.grid_data = params['field_data']

        pygame.init()
        self.screen = pygame.display.set_mode((self.w * self.scale * 2, self.h * self.scale), 0, 32)
        self.canvas = pygame.Surface((self.w * self.scale * 2, self.h * self.scale))
        self.font = pygame.font.SysFont(pygame.font.get_default_font(), 30)
        self.font_q_value = pygame.font.SysFont(pygame.font.get_default_font(), 10)

        rect1 = pygame.Rect(0, 0, self.w * self.scale, self.h * self.scale)
        rect2 = pygame.Rect(self.w * self.scale, 0, self.w * self.scale, self.h * self.scale)

        self.sub1 = self.canvas.subsurface(rect1)
        self.sub2 = self.canvas.subsurface(rect2)
        self.draw_field(params['start_pos'], params['start_direction'])
        self.draw_observed_map(params['start_pos'], params['start_direction'])
        self.top_text = self.font.render(
            'Agent: {}, Targets found: {}, Occupied:{}, Free found: {}, Step: {} '.format(0, 0, 0, 0, 0), False,
            (255, 255, 255))

    def draw_field(self, pos, direction):
        self.sub1.fill(FREE_COLOR)

        # 画边框
        pygame.draw.line(self.sub1, LINE_COLOR, (0, 0), (self.w * self.scale, 0), self.border_width)  # 上
        pygame.draw.line(self.sub1, LINE_COLOR, (0, self.h * self.scale), (self.w * self.scale, self.h * self.scale),
                         self.border_width)  # 下
        pygame.draw.line(self.sub1, LINE_COLOR, (0, 0), (0, self.h * self.scale), self.border_width)  # 左
        pygame.draw.line(self.sub1, DIVIDE_LINE_COLOR, (self.w * self.scale, 0),
                         (self.w * self.scale, self.h * self.scale),
                         self.border_width)  # 右

        # 画棋盘
        for i in range(1, self.w):
            y = i * self.scale
            x = i * self.scale
            startP_row = (0, y)
            endP_row = (self.w * self.scale, y)
            startP_col = (x, 0)
            endP_col = (x, self.h * self.scale)
            pygame.draw.line(self.sub1, LINE_COLOR, startP_row, endP_row, self.line_width)  # 横线
            pygame.draw.line(self.sub1, LINE_COLOR, startP_col, endP_col, self.line_width)  # 竖线

        # draw target
        for i in range(self.h):
            for j in range(self.w):
                if self.grid_data[i, j] == STATEVALUE.TARGET:
                    self.draw_cell(self.sub1, j, i, COLOR_DICT[STATEVALUE.TARGET])
                elif self.grid_data[i, j] == STATEVALUE.OCCUPIED:
                    self.draw_cell(self.sub1, j, i, COLOR_DICT[STATEVALUE.OCCUPIED])

        self.draw_robot(self.sub1, pos, direction)

    def draw_observed_map(self, pos, direction, observed_map=None, new_observed_map=None):
        self.sub2.fill(UNKNOWN_COLOR)

        # 画边框
        pygame.draw.line(self.sub2, LINE_COLOR, (0, 0), (self.w * self.scale, 0), self.border_width)  # 上
        pygame.draw.line(self.sub2, LINE_COLOR, (0, self.h * self.scale), (self.w * self.scale, self.h * self.scale),
                         self.border_width)  # 下
        pygame.draw.line(self.sub2, DIVIDE_LINE_COLOR, (0, 0), (0, self.h * self.scale), self.border_width)  # 左
        pygame.draw.line(self.sub2, LINE_COLOR, (self.w * self.scale, 0), (self.w * self.scale, self.h * self.scale),
                         self.border_width)  # 右

        # 画格子
        for i in range(1, self.w):
            y = i * self.scale
            x = i * self.scale
            startP_row = (0, y)
            endP_row = (self.w * self.scale, y)
            startP_col = (x, 0)
            endP_col = (x, self.h * self.scale)
            pygame.draw.line(self.sub2, LINE_COLOR, startP_row, endP_row, self.line_width)  # 横线
            pygame.draw.line(self.sub2, LINE_COLOR, startP_col, endP_col, self.line_width)  # 竖线

        # draw target
        if observed_map is not None:
            for i in range(self.h):
                for j in range(self.w):
                    if observed_map[i, j] == STATEVALUE.TARGET:
                        self.draw_cell(self.sub2, j, i, COLOR_DICT[STATEVALUE.TARGET])
                    elif observed_map[i, j] == STATEVALUE.FREE:
                        self.draw_cell(self.sub2, j, i, COLOR_DICT[STATEVALUE.FREE])
                    elif observed_map[i, j] == STATEVALUE.OCCUPIED:
                        self.draw_cell(self.sub2, j, i, COLOR_DICT[STATEVALUE.OCCUPIED])

        if new_observed_map is not None:
            for i in range(self.h):
                for j in range(self.w):
                    if new_observed_map[i, j] != STATEVALUE.UNKNOWN:
                        self.draw_cell(self.sub2, j, i, NEW_OBSERVED_SCENE_COLOR)

        self.draw_robot(self.sub2, pos, direction)
        # self.draw_q_value(self.sub2, q_values)
        # self.screen.blit(self.sub2, (self.w * self.scale, 0))

    def draw_q_value(self, q_values):
        sub = self.sub2
        i_len, j_len, a_len = q_values.shape
        for i in range(i_len):
            for j in range(j_len):
                t_up = self.font_q_value.render(str(round(q_values[i][j][0], 2)), False, FONT_COLOR)
                sub.blit(t_up, (i * self.scale, j * self.scale + int(self.scale / 2)))
                t_down = self.font_q_value.render(str(round(q_values[i][j][1], 2)), False, FONT_COLOR)
                sub.blit(t_down, (i * self.scale + self.scale, j * self.scale + int(self.scale / 2)))
                t_left = self.font_q_value.render(str(round(q_values[i][j][2], 2)), False, FONT_COLOR)
                sub.blit(t_left, (i * self.scale + int(self.scale / 2), j * self.scale))
                t_right = self.font_q_value.render(str(round(q_values[i][j][3], 2)), False, FONT_COLOR)
                sub.blit(t_right, (i * self.scale + int(self.scale / 2), j * self.scale + self.scale))

    def draw_cell(self, sub, x, y, cell_color):
        # cell_color = (179, 88, 247)
        # 画标 5 个记点 (400, 400) (240, 240) (240, 560) (560, 240) (560, 560)
        x = x * self.scale + 2
        y = y * self.scale + 2
        position = (x, y, self.scale - 2, self.scale - 2)
        width = 0
        pygame.draw.rect(sub, cell_color, position, width)

    def draw_robot(self, sub, pos, direction):
        x = pos[1] * self.scale + 0.5 * self.scale + 1
        y = pos[0] * self.scale + 0.5 * self.scale + 1

        start_arrow = (x, y)
        end_arrow = (x + 0.5 * self.scale * direction[1], y + 0.5 * self.scale * direction[0])

        pygame.draw.line(sub, ROBOT_COLOR, start_arrow, end_arrow, width=int(0.2 * self.scale))
        pygame.draw.circle(sub, ROBOT_COLOR, (x, y), 0.3 * self.scale - 1, 0)

    def update(self):
        self.screen.blit(self.sub1, (0, 0))

        self.screen.blit(self.sub2, (self.w * self.scale, 0))
        self.screen.blit(self.top_text, (5, 5))

        pygame.display.update()

    def set_top_text(self, step, targets, free, occupied, title=""):
        self.top_text = self.font.render(
            'Agent: {}, Targets found: {}, Occupied:{}, Free found: {}, Step: {} '.format(title, targets,
                                                                                          occupied, free, step), False,
            (255, 255, 255))


if __name__ == '__main__':
    file_path = "../data/ori_data.txt"
    fr = open(file_path, 'rb')
    data = pickle.load(fr)
    width = len(data)
    height = len(data[0])

    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    startPoint = [0, 10]
    startDirection = (-1, 0)

    painter = Painter(width, height, scale=40, grid_data=data, start_pos=startPoint, start_direction=startDirection)
    pygame.display.update()

    pos = np.array(startPoint)
    direction = np.array(startDirection)

    running = True
    # 开启一个事件循环处理发生的事件
    while running:
        # 从消息队列中获取事件并对事件进行处理
        for event in pygame.event.get():
            painter.draw_field(pos, direction)
            painter.draw_observed_map(pos, direction)

            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    direction = directions[0]
                    pos += direction

                elif event.key == pygame.K_DOWN:
                    direction = directions[1]
                    pos += direction

                elif event.key == pygame.K_LEFT:
                    direction = directions[2]
                    pos += direction

                else:
                    direction = directions[3]
                    pos += direction

        pygame.display.update()
