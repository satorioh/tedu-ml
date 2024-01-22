"""
逻辑回归：鸢尾花多分类-交叉验证
"""
import sklearn.datasets as datasets
import pandas as pd
import sklearn.linear_model as linear_model
import sklearn.model_selection as model_selection

iris = datasets.load_iris()

# 将输入和输出整合成一个dataframe，方便画图
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# 整理输入和输出
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 交叉验证
model = linear_model.LogisticRegression(solver='liblinear')
# 传入全部的x/y，5折
scores = model_selection.cross_val_score(model, x, y, cv=5, scoring='f1_weighted')
print('F1-score mean', scores.mean())  # 0.959522933505973, 需要自己手动求平均值
