import numpy as np

array01 = np.arange(1, 10)
array02 = np.arange(9, 0, -1)
print(array01)  # [1 2 3 4 5 6 7 8 9]
print(array02)  # [9 8 7 6 5 4 3 2 1]
print(np.maximum(array01, array02))  # [9 8 7 6 5 6 7 8 9] 取两个数组中的较大值
print(np.minimum(array01, array02))  # [1 2 3 4 5 4 3 2 1] 取两个数组中的较小值
