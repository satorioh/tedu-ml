"""
子图：网格布局
"""
import matplotlib.pyplot as plt

plt.figure('gridspec', facecolor='lightgray')

# 调用GridSpec方法拆分网格式布局
# rows:	行数
# cols:	列数
# 拆分成3行3列的网格对象
gs = plt.GridSpec(3, 3)
# 合并0行的前2列为一个子图表
plt.subplot(gs[0, :2])  # [行，列]
plt.text(0.5, 0.5, '1', ha='center', va='center', fontsize=18)
plt.show()
