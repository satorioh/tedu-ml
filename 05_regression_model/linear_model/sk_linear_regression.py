"""
通过sklearn提供的API实现线性回归
"""
import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv("../../data_test/Salary_Data.csv")
# print(data.head())

# 整理输入（必须二维）和输出（最好是一维）
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 创建模型
model = lm.LinearRegression()  # 支持多个权重 w1x1 + w2x2 + ... + b
# 训练模型
model.fit(x.values, y.values)
print(f"model.coef_:", model.coef_)
print(f"model.intercept_:", model.intercept_)
# 预测
pred_y = model.predict(x.values)
#
# 画回归模型线和实际点
plt.scatter(x, y, label="Real")
plt.plot(x, pred_y, label="Predict", color="orangered")
plt.legend()
plt.show()

# 模型评估
# 从全部数据中抽取一部分数据作为测试集(假设测试集没参加过训练)
test_x = x.iloc[::4].values
test_y = y[::4].values
pred_test_y = model.predict(test_x)

# # MAE
print("MAE:", sm.mean_absolute_error(test_y, pred_test_y))  # 4587.366522327396

# MSE
print("MSE:", sm.mean_squared_error(test_y, pred_test_y))  # 29784216.41962167

# RMSE
print("RMSE:", np.sqrt(sm.mean_squared_error(test_y, pred_test_y)))  # 5457.491770000361

# 中位数绝对误差
print("MdAE:", sm.median_absolute_error(test_y, pred_test_y))  # 4895.445366109856

# R2得分
print("R2:", sm.r2_score(test_y, pred_test_y))  # 0.9645484959659238

# 保存模型
with open("linear_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("模型保存成功！")
