"""
波士顿房价预测：使用单棵决策树回归模型
506个样本，13列特征+1个价格中位数
"""
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics
import sklearn.tree as tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_df = pd.read_csv("../../data_test/boston.txt", sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
print(data.shape)
print(target.shape)

def get_feature_names(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    start_index = lines.index(' Variables in order:\n') + 1
    end_index = start_index + 13
    features = [line.strip().split()[0] for line in lines[start_index:end_index] if line.strip()]
    return features


feature_names = get_feature_names('../../data_test/boston.txt')
print(feature_names)

x = data
y = target
# 划分训练集和测试集（训练集数据不能有顺序）
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=7)

# 单棵决策树回归
model = tree.DecisionTreeRegressor(max_depth=6, random_state=7)
model.fit(x_train, y_train)
lm_pred_y_train = model.predict(x_train)
lm_pred_y_test = model.predict(x_test)
print("训练集MSE：", metrics.mean_squared_error(y_train, lm_pred_y_train))
print("测试集MSE：", metrics.mean_squared_error(y_test, lm_pred_y_test))
print("训练集R2：", metrics.r2_score(y_train, lm_pred_y_train))
print("测试集R2：", metrics.r2_score(y_test, lm_pred_y_test))
"""
训练集MSE： 4.552430860318175
测试集MSE： 27.360074525649082
训练集R2： 0.9466499293522961
测试集R2： 0.6614114488851882
"""

# 特征重要性
fis = pd.Series(model.feature_importances_, index=feature_names)
fis = fis.sort_values(ascending=False)
print(fis)
plt.bar(fis.index, fis.values)
# 画决策树
tree.plot_tree(model, fontsize=10, feature_names=feature_names, filled=True, rounded=True)
plt.show()
