"""
随机森林分类树：学习曲线
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

# 学习曲线
train_sizes = np.arange(0.1, 1.1, 0.1)
train_size, train_score, test_score = model_selection.learning_curve(model, train_x, train_y, train_sizes=train_sizes,
                                                                     cv=5)
avg_score = test_score.mean(axis=1)
plt.plot(train_sizes, avg_score, 'o-', label='Learning Curve')
plt.legend()
plt.show()
