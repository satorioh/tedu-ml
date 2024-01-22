import numpy as np

a = np.arange(1, 51).reshape(5, 10)
print(a)
# 获取所有记录，不要最后一列，返回二维数组
print(a[:, :-1])

# 获取所有记录的最后一列，返回一维数组
print(a[:, -1])  # [10 20 30 40 50]
