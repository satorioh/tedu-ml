"""
使用网格搜索实现svm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
import sklearn.svm as svm
import sklearn.metrics as metrics

data = pd.read_csv('../data_test/multiple2.txt', header=None, names=['x1', 'x2', 'y'])
# 整理输入和输出
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分训练集和测试集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=7, test_size=0.2, stratify=y)

# 构建模型
sub_model = svm.SVC()
params_grid = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
               {'kernel': ['poly'], 'C': [1], 'degree': [2, 3]},
               {'kernel': ['rbf'], 'C': [1, 10, 100], 'gamma': [1, 0.1, 0.01]}]
model = model_selection.GridSearchCV(sub_model, params_grid, cv=3)

# 训练模型
model.fit(x, y)

print("最优参数：", model.best_params_)
print("最优得分：", model.best_score_)
best_model = model.best_estimator_

# 预测
pred_y = best_model.predict(x_test)
# 评估
print(metrics.classification_report(y_test, pred_y))
