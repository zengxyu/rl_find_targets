import pickle

import numpy as np

grid_data = np.loadtxt("ori_data2.txt")
grid_data = np.array(grid_data)
print(grid_data)
fw = open('grid_data2.txt', 'wb')
pickle.dump(grid_data, fw)

