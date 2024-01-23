"""
凝聚层次聚类算法
"""
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster

data = pd.read_csv("../data_test/multiple3.txt", header=None, names=["x1", "x2"])

# 构建模型
model = cluster.AgglomerativeClustering(n_clusters=4)
# 训练模型
model.fit(data.values)
# 获取分类结果
labels = model.labels_
print(labels)
"""
[1 1 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1
 3 0 2 1 3 0 2 1 3 0 2 1 3 0 0 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 1
 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 0 3 0 2 1 3 0 2 1 3 0 2 1 3 0
 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 1 0 2
 1 1 0 2 1 3 0 2 1 3 0 3 1 3 0 2 1 3 0 2 1 1 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1
 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2]
"""

# 画图，显示聚类结果
plt.scatter(data["x1"], data["x2"], c=labels, cmap="viridis")
plt.colorbar()
plt.show()
