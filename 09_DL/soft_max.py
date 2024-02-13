"""
softmax激活函数
"""
import numpy as np


def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)  # 防止溢出
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y


a = np.array([0.3, 2.9, 4.0])
print(softmax(a))  # [0.01821127 0.24519181 0.73659691]
