"""
多项式回归.
1.扩展特征(增加高次项)
2.将扩展完特征的数据交给线性回归解析
"""
import pandas as pd
import sklearn.linear_model as lm
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt

data = pd.read_csv("../../data_test/Salary_Data.csv")

# 整理输入和输出
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 构建多项式回归模型
model = pl.make_pipeline(sp.PolynomialFeatures(3), lm.LinearRegression())
# 训练模型
model.fit(x.values, y.values)
# 预测
pred_y = model.predict(x.values)
# 画回归线
plt.plot(x, pred_y, color="orangered", label="Predict Poly Regression")
plt.scatter(x, y, color="dodgerblue", label="Samples")
plt.legend()
plt.show()
