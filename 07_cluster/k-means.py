"""
k-means聚类算法
"""
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster

data = pd.read_csv("../data_test/multiple3.txt", header=None, names=["x1", "x2"])

# 画图来确定n_clusters的值
# plt.scatter(data["x1"], data["x2"])
# plt.show()

# 构建模型
model = cluster.KMeans(n_clusters=4, random_state=0)
# 训练模型
model.fit(data.values)
# 预测结果
labels = model.labels_
print(labels)
"""
[3 3 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 0 0 2 1 3 0 2 1 3 0 2 1 3
 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0
 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 2 0 2 1 3 0 2 1 3 0 2 1 3 0 2
 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1
 3 3 2 1 3 0 2 1 3 0 2 0 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1 3
 0 2 1 3 0 2 1 3 0 2 1 3 0 2 1]
"""

# 查看预测结果的中心
centers = model.cluster_centers_
print(centers)
"""
[[3.1428     5.2616    ]
 [7.07326531 5.61061224]
 [5.91196078 2.04980392]
 [1.831      1.9998    ]]
"""

# 对输入的实例进行预测
pred_y = model.predict([[1.1, 2.2]])
print("pred_y", pred_y) # pred_y [3]

# 画图，显示聚类结果
plt.scatter(data["x1"], data["x2"], c=labels, cmap="viridis")
plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)
plt.colorbar()
plt.show()
