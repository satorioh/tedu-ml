"""
支持向量机：找到最大间隔超平面
线性核函数
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
import sklearn.svm as svm
import sklearn.metrics as metrics

data = pd.read_csv('../data_test/multiple2.txt', header=None, names=['x1', 'x2', 'y'])

# 画出数据点，观察是否线性可分
# plt.scatter(data.x1, data.x2, c=data.y, cmap='bwr')
# plt.colorbar()
# plt.show()

# 整理输入和输出
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分训练集和测试集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=7, test_size=0.2, stratify=y)

# 构建模型
model = svm.SVC(kernel='linear')
# 训练模型
model.fit(x_train, y_train)
# 预测
pred_y = model.predict(x_test)
# 评估
print(metrics.classification_report(y_test, pred_y))
"""
 precision    recall  f1-score   support

           0       0.60      0.93      0.73        30
           1       0.85      0.37      0.51        30

    accuracy                           0.65        60
   macro avg       0.72      0.65      0.62        60
weighted avg       0.72      0.65      0.62        60
"""

# 暴力绘制分类边界
# 1.将x1的最小值到x1的最大值拆分200个数
x1s = np.linspace(x.x1.min(), x.x1.max(), 200)
# 2.将x2的最小值到x2的最大值拆分200个数
x2s = np.linspace(x.x2.min(), x.x2.max(), 200)
# 3.组合x1和x2的所有情况，4w个点
points = []
for x1 in x1s:
    for x2 in x2s:
        points.append([x1, x2])
points = pd.DataFrame(points, columns=['x1', 'x2'])
# 4.将4w个点带入模型中，得到预测类别
points_label = model.predict(points)
# 5.将4w个点画散点图，颜色根据预测类别的变化而变化
plt.scatter(points.x1, points.x2, c=points_label, cmap='gray')
# 6.将样本的散点图画出来
plt.scatter(data.x1, data.x2, c=data.y, cmap='bwr')
plt.show()
