"""
二值化
根据一个事先给定的阈值，用0和1来表示特征值是否超过阈值
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([[65.5, 89.0, 73.0],
                        [55.0, 99.0, 98.5],
                        [45.0, 22.5, 60.0]])

copy_sample = raw_samples.copy()

# 将及格的数据转为1,不及格的数据转为0
# 阈值:60
# 拿到大于等于60的数据，赋值为1
# 拿到小于60的数据，赋值为0
res = np.where(copy_sample >= 60, 1.0, 0.0)
print(res)
"""
[[1. 1. 1.]
 [0. 1. 1.]
 [0. 0. 1.]]
"""

# 基于skLearn提供的API实现
binarizer = sp.Binarizer(threshold=59.9)
result = binarizer.transform(raw_samples)
print(result)
