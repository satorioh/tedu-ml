"""
梯度提升树(Gradient Boosting Decision Tree, GBDT)
"""
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import pandas as pd
import numpy as np

raw_df = pd.read_csv("../../data_test/boston.txt", sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


def get_feature_names(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    start_index = lines.index(' Variables in order:\n') + 1
    end_index = start_index + 13
    features = [line.strip().split()[0] for line in lines[start_index:end_index] if line.strip()]
    return features


feature_names = get_feature_names('../../data_test/boston.txt')

x = data
y = target
# 划分训练集和测试集（训练集数据不能有顺序）
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=7)

# 构建GBDT
model = ensemble.GradientBoostingRegressor(max_depth=3, n_estimators=400, random_state=7)  # n_estimators：弱学习器数量
model.fit(x_train, y_train)
lm_pred_y_train = model.predict(x_train)
lm_pred_y_test = model.predict(x_test)
print("训练集MSE：", metrics.mean_squared_error(y_train, lm_pred_y_train))
print("测试集MSE：", metrics.mean_squared_error(y_test, lm_pred_y_test))
print("训练集R2：", metrics.r2_score(y_train, lm_pred_y_train))
print("测试集R2：", metrics.r2_score(y_test, lm_pred_y_test))
"""
训练集MSE： 0.1503082750039688
测试集MSE： 11.904781329119633
训练集R2： 0.9982385329208853
测试集R2： 0.8526750116200696
"""
