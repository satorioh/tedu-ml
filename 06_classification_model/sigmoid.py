"""
伪造一组预测值，画sigmoid函数图像
"""
import numpy as np
import matplotlib.pyplot as plt

pred_y = np.linspace(-10, 10, 100)
sigmoid_y = 1 / (1 + np.exp(-pred_y))

plt.plot(pred_y, sigmoid_y)
plt.show()
