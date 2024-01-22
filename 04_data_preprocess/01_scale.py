"""
均值移除：将特征值减去其均值（使得新的均值为0），然后除以其标准差（使得新的标准差为1）
目的：将所有特征值调整到同一尺度
"""
import numpy as np
import sklearn.preprocessing as sp

raw_sample = np.array([[3.0, -100.0, 2000.0],
                       [0.0, 400.0, 3000.0],
                       [1.0, -400.0, 2000.0]])

std_sample = raw_sample.copy()

# 1.减去当前列的平均值
# 2.离差/原始数据的标准差
for col in std_sample.T:
    col_mean = col.mean()  # 平均值
    col_std = col.std()  # 标准差
    col -= col_mean
    col /= col_std

print(std_sample)
"""
[[ 1.33630621 -0.20203051 -0.70710678]
 [-1.06904497  1.31319831  1.41421356]
 [-0.26726124 -1.1111678  -0.70710678]]
"""

print(std_sample.mean(axis=0))  # 查看每一列（垂直方向）的均值是否为0（近似0）
"""
[ 5.55111512e-17  0.00000000e+00 -2.96059473e-16]
"""

print(std_sample.std(axis=0))  # 查看每一列（垂直方向）的标准差是否为1
"""
[1. 1. 1.]
"""
print("==" * 20)

# 基于skLearn提供的API实现均值移除
scaler = sp.StandardScaler()
res = scaler.fit_transform(raw_sample)
print(res)
"""
[[ 1.33630621 -0.20203051 -0.70710678]
 [-1.06904497  1.31319831  1.41421356]
 [-0.26726124 -1.1111678  -0.70710678]]
"""
print(res.mean(axis=0))
"""
[ 5.55111512e-17  0.00000000e+00 -2.96059473e-16]
"""
print(res.std(axis=0))
"""
[1. 1. 1.]
"""
