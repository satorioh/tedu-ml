"""
朴素贝叶斯分类器
"""
import pandas as pd
import numpy as np
import sklearn.model_selection as model_selection
import sklearn.naive_bayes as naive_bayes
import sklearn.metrics as metrics

data = pd.read_csv('../data_test/multiple1.txt', header=None, names=['x1', 'x2', 'y'])

# 整理输入和输出
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分训练集和测试集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=7, test_size=0.2, stratify=y)

# 构建模型
model = naive_bayes.GaussianNB()
# 训练模型
model.fit(x_train, y_train)
# 预测
pred_y = model.predict(x_test)
# 评估
print(metrics.classification_report(y_test, pred_y))