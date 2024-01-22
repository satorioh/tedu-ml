"""
读取SalaryData2.csv
根据工作经验预测薪资(构建线性模型)
并绘制回归线
"""
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.metrics as sm

data = pd.read_csv("../../data_test/Salary_Data2.csv")
# print(data.head())
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 构建线性模型
model = lm.LinearRegression()
model.fit(x.values, y.values)
pred_y = model.predict(x.values)

# 构建岭回归模型
model_ridge = lm.Ridge(alpha=98)
model_ridge.fit(x.values, y.values)
pred_y_ridge = model_ridge.predict(x.values)


plt.plot(x, pred_y, label="Linear Predict", color="orangered")
plt.plot(x, pred_y_ridge, label="Ridge Predict", color="green")
plt.scatter(x, y, label="Real")
plt.legend()
plt.show()

# 寻找岭回归alpha最优参数
# 找到一组测试集，假设没参加过训练
test_x = x.iloc[:30:4]
test_y = y[:30:4]
alpha_list = np.arange(90, 101, 1)
score_list = []

for alpha in alpha_list:
    model_ridge_test = lm.Ridge(alpha=alpha)
    model_ridge_test.fit(x.values, y.values)
    pred_y_ridge_test = model_ridge_test.predict(test_x.values)
    score = sm.r2_score(test_y.values, pred_y_ridge_test)
    print(f"alpha={alpha}, score={score}")
    score_list.append(score)

result = pd.Series(score_list, index=alpha_list)
print(f"最优alpha={result.idxmax()}, 最优score={result.max()}")
