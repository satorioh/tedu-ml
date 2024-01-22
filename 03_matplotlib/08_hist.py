"""
直方图
X轴表示数据的范围，Y轴表示频率
"""
import matplotlib.pyplot as plt
import numpy as np

data = np.random.normal(175, 5, 20000)
print(data)

plt.hist(data, bins=100)
plt.show()
