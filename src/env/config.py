import os
import pickle
from enum import IntEnum
import numpy as np


class STATEVALUE(IntEnum):
    UNKNOWN = 0,
    FREE = 1,
    TARGET = 2
    OCCUPIED = 3,
    ROBOT = 4,


class ACTION(IntEnum):
    MOVE_UP = 0,
    MOVE_DOWN = 1,
    MOVE_LEFT = 2,
    MOVE_RIGHT = 3


MOVEMENTS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


class REWARD(IntEnum):
    FREE = 0
    TARGET = 1
    OCCUPIED = -1


class FIELD:
    # data
    grid_path = os.path.join("src/data", "grid_data.txt")
    if os.path.exists(grid_path):
        data = pickle.load(open(grid_path, 'rb'))
        h = data.shape[0]
        w = data.shape[1]
    else:
        print("file:{} not exist!".format(grid_path))


if __name__ == '__main__':
    for a in range(len(ACTION)):
        print(a)
