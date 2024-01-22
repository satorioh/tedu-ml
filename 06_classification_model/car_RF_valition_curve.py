"""
随机森林分类树：验证曲线
"""
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
import sklearn.ensemble as ensemble
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt

data = pd.read_csv('../data_test/car.txt', header=None)
# print(data[0].unique())

encoders = {}
train_data = pd.DataFrame()

for col in data.columns:
    encoder = preprocessing.LabelEncoder()
    encoders[col] = encoder
    train_data[col] = encoder.fit_transform(data[col])

# 整理输入和输出
train_x = train_data.iloc[:, :-1]
train_y = train_data.iloc[:, -1]

# 构建随机森林
model = ensemble.RandomForestClassifier(max_depth=6, n_estimators=400, random_state=7)

# 验证曲线
param_range = np.arange(100, 1001, 100)
train_scores1, test_scores1 = model_selection.validation_curve(model, train_x, train_y, param_name='n_estimators', cv=5,
                                                               param_range=param_range)
avg_test_scores = test_scores1.mean(axis=1)
plt.plot(param_range, avg_test_scores, 'o-', label='test')
plt.show()
