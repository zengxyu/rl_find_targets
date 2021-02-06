import pickle
import numpy as np

from src.env.config import STATEVALUE

w = 40
h = 40
grid = np.zeros((w, h), dtype=int)

for i in range(w):
    for j in range(h):
        # if i in np.arange(4, 10) and j in np.arange(4, 10):
        #     grid[i][j] = STATEVALUE.TARGET
        # elif i in np.arange(9, 11) and j in np.arange(9, 11):
        #     grid[i][j] = STATEVALUE.TARGET
        # elif i in np.arange(3, 8) and j in np.arange(24, 28):
        #     grid[i][j] = STATEVALUE.TARGET
        # elif i in np.arange(10, 15) and j in np.arange(36, 39):
        #     grid[i][j] = STATEVALUE.TARGET
        # elif i in np.arange(16, 18) and j in np.arange(2, 5):
        #     grid[i][j] = STATEVALUE.TARGET
        # elif i in np.arange(17, 19) and j in np.arange(10, 19):
        #     grid[i][j] = STATEVALUE.TARGET
        # elif i in np.arange(20, 22) and j in np.arange(20, 24):
        #     grid[i][j] = STATEVALUE.TARGET
        # elif i in np.arange(17, 19) and j in np.arange(26, 29):
        #     grid[i][j] = STATEVALUE.TARGET
        # elif i in np.arange(22, 27) and j in np.arange(15, 17):
        #     grid[i][j] = STATEVALUE.TARGET
        # elif i in np.arange(25, 28) and j in np.arange(4, 10):
        #     grid[i][j] = STATEVALUE.TARGET
        # elif i in np.arange(26, 29) and j in np.arange(23, 28):
        #     grid[i][j] = STATEVALUE.TARGET
        # elif i in np.arange(32, 38) and j in np.arange(19, 21):
        #     grid[i][j] = STATEVALUE.TARGET
        # elif i in np.arange(30, 34) and j in np.arange(20, 27):
        #     grid[i][j] = STATEVALUE.TARGET
        # elif i in np.arange(35, 38) and j in np.arange(35, 38):
        #     grid[i][j] = STATEVALUE.TARGET
        # else:
        grid[i][j] = STATEVALUE.FREE

for i in range(w):
    string = ""
    for j in range(h):
        if grid[i][j] == 1:
            string += "%03s" % "1"
        else:
            string += "%03s" % "0"
    print(string)
grid = grid.astype(np.int)
fw = open('ori_data.txt', 'wb')
np.savetxt("ori_data.txt", grid, fmt='%i')
pickle.dump(grid, fw)
fw.close()
fr = open('ori_data.txt', 'rb')
data = pickle.load(fr)
print(data)
