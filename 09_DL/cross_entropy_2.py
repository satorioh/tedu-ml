import numpy as np


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# 假设当前某样本真实类别为3， softmax输出一共有5个类别
y_true = [0, 0, 0, 1, 0]  # 真实概率：0类别概率为0， 1类别概率为0， 2类别概率为0， 3类别概率为1， 4类别概率为0
y1_pred = [0.1, 0.1, 0.1, 0.6, 0.1]  # 预测概率：0类别概率为0.1， 1类别概率为0.1， 2类别概率为0.1， 3类别概率为0.6， 4类别概率为0.1

print("交叉熵1：", cross_entropy_error(np.array(y1_pred), np.array(y_true)))  # 0.510825457099338
