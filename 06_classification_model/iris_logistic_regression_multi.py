"""
逻辑回归：鸢尾花多分类
"""
import sklearn.datasets as datasets
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics

iris = datasets.load_iris()
# print(iris.keys())
# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data.shape)
# print(iris.target)

# 将输入和输出整合成一个dataframe，方便画图
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# 萼片长度和宽度的可视化
plt.scatter(data['sepal length (cm)'], data['sepal width (cm)'], c=data['target'], cmap='brg')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.colorbar(ticks=[0, 1, 2], label='target')
# plt.show()

# 花瓣长度和宽度的可视化
plt.figure()
plt.scatter(data['petal length (cm)'], data['petal width (cm)'], c=data['target'], cmap='brg')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.colorbar(ticks=[0, 1, 2], label='target')
# plt.show()

# 整理输入和输出
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分训练集和测试集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=7, stratify=y)
# 构建逻辑回归模型
model = linear_model.LogisticRegression(solver='liblinear')
# 训练模型
model.fit(x_train, y_train)
# 预测
y_pred = model.predict(x_test)
# 评估
print('真实结果：', y_test.values)  # [2 1 0 1 2 0 1 1 0 1 1 1 0 2 0 1 2 2 0 0 1 2 1 2 2 2 1 1 2 2]
print('预测结果：', y_pred)  # [2 2 0 2 2 0 1 1 0 1 2 2 0 2 0 2 2 2 0 0 1 2 1 2 1 2 1 1 2 2]
print('精度：', metrics.accuracy_score(y_test, y_pred))  # 0.8
print('查准率', metrics.precision_score(y_test, y_pred, average='macro'))  # 0.8472222222222222
print('查全率', metrics.recall_score(y_test, y_pred, average='macro'))  # 0.8308080808080809
print('F1-score', metrics.f1_score(y_test, y_pred, average='macro'))  # 0.8230769230769232
print('混淆矩阵：\n', metrics.confusion_matrix(y_test, y_pred))
print('分类报告：\n', metrics.classification_report(y_test, y_pred))
