"""
自定义窗口
"""
import numpy as np
import matplotlib.pyplot as plt

plt.figure(
    'custom-fig',  # 窗口标题栏文本
    figsize=(6, 6),  # 窗口大小 <元组>
    facecolor='lightgray'  # 图表背景色
)
plt.title('my title', fontsize=24)
plt.xlabel('xxx', fontsize=12)
plt.ylabel('yyy', fontsize=12)

# 设置图表网格线  linestyle设置网格线的样式
#	-  or solid 粗线
#   -- or dashed 虚线
#   -. or dashdot 点虚线
#   :  or dotted 点线
plt.grid(linestyle='-.')
plt.tight_layout()
x = [1, 2, 3]
y = [1, 2, 3]
plt.plot(x, y)

plt.show()
