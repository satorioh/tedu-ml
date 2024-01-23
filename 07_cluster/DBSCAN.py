"""
DBSCAN（噪声密度）聚类算法
"""
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster

data = pd.read_csv("../data_test/multiple3.txt", header=None, names=["x1", "x2"])

# 画图来确定n_clusters的值
# plt.scatter(data["x1"], data["x2"])
# plt.show()

# 构建模型
model = cluster.DBSCAN(eps=0.65, min_samples=5)  # eps: 半径， min_samples: 最小样本数
# 训练模型
model.fit(data.values)
# 获取分类结果
labels = model.labels_
print(labels)
"""
-1是噪声点
[-1  1 -1  0  1  3  2  0  1  3  2  0  1  3  2  0  1  3  2 -1  1  3  2  0
 -1  3 -1  0  1  3  2  0  1  3  2  0  1  3  2  0  1  3  2  0  1  3 -1  0
  1  3  2  0  1  3  2  0  1  3  2  0  1  3  2  0  1  3  2  0  1  3  2  0
  1 -1  2  0  1  3  2  0  1  3  2  0  1  3  2  0  1  3  2  0  1  3  2  0
 -1  3  2  0  1  3 -1  0  1  3  2 -1  1  3  2  0  1  3 -1  0 -1  3  2  0
  1  3  2 -1  1  3 -1  0  1  3  2 -1  1  3  2  0 -1  3  2  0  1  3  2 -1
  1 -1  2  0 -1  1  2  0  1  3  2  0  1  3 -1 -1  1  3  2  0 -1  3  2  0
  1 -1  2  0  1  3  2  0  1  3  2  0  1  3  2  0  1  3  2  0  1  3  2  0
  1  3 -1  0  1  3  2  0]
"""

# 画图，显示聚类结果
plt.scatter(data["x1"], data["x2"], c=labels, cmap="viridis")
plt.colorbar()
plt.show()
