"""
柱状图
"""

import matplotlib.pyplot as plt
import numpy as np

# 绘制苹果12个月的销量
x = np.arange(1, 13)
apples = np.random.normal(30000, 2000, 12)
oranges = np.random.normal(30000, 2000, 12)

plt.bar(x - 0.2, apples, width=0.4, label='Apple')
plt.xticks(x)
plt.bar(x + 0.2, oranges, width=0.4, label='Orange')

# for i in range(len(x)):
#     plt.text(x[i], apples[i], int(apples[i]), ha='center', va='bottom')
plt.legend()
plt.tight_layout()
plt.show()
