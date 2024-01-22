"""
散点图
"""
import matplotlib.pyplot as plt
import numpy as np

n = 2000
# 175:	期望值  : 均值
# 10:	标准差 : 震荡幅度
# n:	数字生成数量
height = np.random.normal(175, 10, n)
weight = np.random.normal(70, 10, n)

plt.scatter(height, weight, c=height, cmap='jet')
plt.colorbar()
plt.show()
