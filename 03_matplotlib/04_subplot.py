"""
子图：矩阵式布局
"""
import matplotlib.pyplot as plt

plt.figure("subplot", facecolor="lightgray")

# for i in range(1, 10):
#     plt.subplot(3, 3, i)
#     plt.plot([1, 2, 3], [1, 2, 3])
#     plt.plot([1, 2, 3], [3, 2, 1])


for i in range(1, 10):
    plt.subplot(3, 3, i)
    plt.text(0.5, 0.5, i, fontsize=20, ha='center', va='center')
    # 去除刻度
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()

plt.show()
