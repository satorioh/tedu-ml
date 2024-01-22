"""
逻辑回归：鸢尾花二分类
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

# 从全部数据中挑选两个类别（1和2），进行二分类
sub_data = data[data['target'] != 0]
print(sub_data)
# 整理输入和输出
x = sub_data.iloc[:, :-1]
y = sub_data.iloc[:, -1]

# 划分训练集和测试集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=7)
# 构建逻辑回归模型
model = linear_model.LogisticRegression(solver='liblinear')
# 训练模型
model.fit(x_train, y_train)
# 预测
y_pred = model.predict(x_test)
# 评估
print('真实结果：', y_test.values)  # [1 1 2 2 1 1 2 2 2 1 1 1 2 1 2 1 1 2 1 1]
print('预测结果：', y_pred)  # [1 1 2 2 1 1 2 2 2 2 1 1 2 1 2 1 1 2 2 2]
print('准确率：', metrics.accuracy_score(y_test, y_pred))  # 0.85
