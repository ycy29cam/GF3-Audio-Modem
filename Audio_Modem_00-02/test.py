import numpy as np

test = np.array([[2, 2], [3, 3], [5, 5], [2, 2] ,[3, 3]])
pilot_index = [0, 1, 3, 4]

Y = np.mean(np.array([test[i] for i in pilot_index]), axis=0)
print(Y)
