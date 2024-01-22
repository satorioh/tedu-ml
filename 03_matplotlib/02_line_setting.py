import numpy as np
import matplotlib.pyplot as plt

# 画出-π到+π之间的正弦函数图像
xs = np.linspace(-np.pi, np.pi, 200)
sin_x = np.sin(xs)
cos_x = np.cos(xs) / 2
plt.plot(xs, sin_x, linestyle="--", linewidth=3, color="red", label=r'$y=sin(x)$')
plt.plot(xs, cos_x, linestyle="-.", linewidth=5, alpha=0.5, label=r'$y=\frac{1}{2}cos(x)$')

# 坐标轴范围（第一象限）
# plt.xlim(0, np.pi + 0.1)
# plt.ylim(0, 1 + 0.1)

# 坐标轴刻度
plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
           [r'-$\pi$', r'-$\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])

# 设置坐标轴
ax = plt.gca()  # 拿到当前的坐标系
# 获取其中某个坐标轴
top = ax.spines['top']
right = ax.spines['right']
left = ax.spines['left']
bottom = ax.spines['bottom']

# 设置坐标轴的颜色
# color: <str> 颜色值字符串
top.set_color('none')
right.set_color('none')

"""
# 设置坐标轴的位置。 该方法需要传入2个元素的元组作为参数
# type: <str> 移动坐标轴的参照类型  一般为'data' (以数据的值作为移动参照值)
# val:  参照值
"""
left.set_position(('data', 0))
bottom.set_position(('data', 0))

# 图例
plt.legend()

# 绘制点
plt.scatter([-np.pi / 2, np.pi / 2], [-1, 1], s=100, zorder=3, facecolor='green')

plt.show()
