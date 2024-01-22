"""
正则化
用每个样本的每个特征值，除以该样本各个特征值绝对值之和（L1）.
变换后的样本矩阵，每个样本的特征值绝对值之和为1
"""
import numpy as np
import sklearn.preprocessing as sp

raw_sample = np.array([[10.0, 20.0, 5.0],
                       [8.0, 10.0, 1.0]])

copy_sample = raw_sample.copy()
for col in copy_sample:
    col /= abs(col).sum()

print(copy_sample)
"""
[[0.28571429 0.57142857 0.14285714]
 [0.42105263 0.52631579 0.05263158]]
"""

print(copy_sample.sum(axis=1))  # 每行（水平方向）和为1
"""
[1. 1.]
"""

# 基于skLearn提供的API实现
normalizer = sp.Normalizer(norm='l1')  # 默认norm=l2
res = normalizer.fit_transform(raw_sample)
print(res)
"""
[[0.28571429 0.57142857 0.14285714]
 [0.42105263 0.52631579 0.05263158]]
"""
