import numpy as np
import pandas as pd
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import matplotlib.pyplot as plt
from sklearn import ensemble
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics

data = pd.read_csv("../data_test/advertising.csv")

# 整理输入和输出
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 拆分训练集和测试集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=7)

# 构建网格搜索模型
params_grid = {
    'max_depth': np.arange(4, 18),
    'n_estimators': np.arange(400, 1001, 100)
}
sub_model = ensemble.RandomForestRegressor(random_state=7)
model = model_selection.GridSearchCV(sub_model, params_grid, cv=3)

# 构建多项式回归模型
# model = pipeline.make_pipeline(preprocessing.PolynomialFeatures(5), linear_model.LinearRegression())
# 构建随机森林
# model = ensemble.RandomForestRegressor(max_depth=10, n_estimators=1000, random_state=7)


# 训练模型
model.fit(x_train.values, y_train.values)

print("最优参数：", model.best_params_)
print("最优得分：", model.best_score_)
best_model = model.best_estimator_

# 预测
pred_y = best_model.predict(x_test.values)
# 评估
# 均方误差和R2得分
print("测试集均方误差:", metrics.mean_squared_error(y_test.values, pred_y))
print("测试集R2得分:", metrics.r2_score(y_test.values, pred_y))
"""
测试集均方误差: 1.2651391864808308
测试集R2得分: 0.9593528844054271
"""
test_data = [[250, 50, 50]]
pred_y_test = best_model.predict(test_data)
print("预测结果:", pred_y_test)  # [8.82994167]
