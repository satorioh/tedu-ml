"""
基于Python代码实现梯度下降求模型参数
流程：
使用线性回归模型（一次函数/直线：y = w1*x + w0）做拟合，我们可以使用MSE（均方差）作为loss function，然后求MSE的极小值
如何求MSE的极小值？使用梯度下降法，不断迭代w1和w0，来观察loss值的变化
如何使用梯度下降法来迭代w？涉及到参数更新公式： w = w - η*（loss function关于w的偏导数）
通过w1和w0的迭代，观察loss值的变化，最终确定模型（y = w1*x + w0）的取值
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])  # 输入集
y = np.array([5.0, 5.5, 6.0, 6.8, 7.1])  # 输出集

# 画散点图查看是否有线性关系
# plt.scatter(x, y)
# plt.show()

# 目标函数：y = w1 * x + w0
w1 = 1  # 权重，一般为随机数但不能为0，此处为了方便演示
w0 = 1  # 偏置，一般初始为0 Or 1
learning_rate = 0.01  # 学习率，不能太大也不能太小
epoch = 300  # 迭代次数
epochs = []  # 记录迭代次数
losses = []  # 记录损失值
w1s = []  # 记录w1的变化
w0s = []  # 记录w0的变化


def get_loss(w, b):
    """
    计算损失值
    :param w: 权重
    :param b: 偏置
    :return: 损失值
    """
    return ((w * x + b - y) ** 2).sum() / 2


# 开启迭代，更新参数
for i in range(epoch):
    epochs.append(i)  # 记录迭代次数
    loss = get_loss(w1, w0)  # 计算损失值
    losses.append(loss)  # 记录损失值
    w1s.append(w1)  # 记录w1的变化
    w0s.append(w0)  # 记录w0的变化
    print(f"epoch={i}, w1={w1}, w0={w0}, loss={loss}")
    d0 = (w0 + w1 * x - y).sum()  # 求w0的梯度
    d1 = (x * (w1 * x + w0 - y)).sum()  # 求w1的梯度
    # 根据参数更新公式来更新参数
    w0 = w0 - learning_rate * d0
    w1 = w1 - learning_rate * d1

print(f"最终参数：w0={w0}, w1={w1}")

# 画模型线
# pred_y = w1 * x + w0  # 预测值
# plt.plot(x, pred_y, color='orangered')
# plt.scatter(x, y)  # 画散点图

# 画损失值和参数的变化曲线
plt.subplot(3, 1, 1)
plt.plot(epochs, w0s, color="dodgerblue", label="w0")  # 画w0的变化曲线
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(epochs, w1s, color="dodgerblue", label="w1")  # 画w1的变化曲线
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(epochs, losses, color="orangered", label="loss")  # 画损失值的变化曲线
plt.legend()
plt.show()
