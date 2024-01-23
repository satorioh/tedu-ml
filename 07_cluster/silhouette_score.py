"""
轮廓系数（聚类算法评价指标）
"""
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as metrics

data = pd.read_csv("../data_test/multiple3.txt", header=None, names=["x1", "x2"])

# 构建模型
model = cluster.AgglomerativeClustering(n_clusters=4)
# 训练模型
model.fit(data.values)
# 获取分类结果
labels = model.labels_

# 画图，显示聚类结果
plt.scatter(data["x1"], data["x2"], c=labels, cmap="viridis")
plt.colorbar()
plt.show()

# 评估聚类结果
# 轮廓系数
print(metrics.silhouette_score(data, labels, sample_size=len(data), metric='euclidean'))  # 0.5736608796903743
