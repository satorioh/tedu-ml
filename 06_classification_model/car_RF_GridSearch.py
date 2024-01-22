"""
随机森林分类树：网格搜索
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

# 构建网格搜索模型
params_grid = {
    'max_depth': np.arange(4, 18),
    'n_estimators': np.arange(100, 801, 50)
}
sub_model = ensemble.RandomForestClassifier(random_state=7)
model = model_selection.GridSearchCV(sub_model, params_grid, cv=3)

# 训练模型
model.fit(train_x, train_y)
# 准备测试数据
test_data = [['high', 'med', '5more', '4', 'big', 'low', 'unacc'],
             ['high', 'high', '4', '4', 'med', 'med', 'acc'],
             ['low', 'low', '2', '4', 'small', 'high', 'good'],
             ['low', 'med', '3', '4', 'med', 'high', 'vgood']]

# 将测试数据转换为数字
test_data = pd.DataFrame(test_data)
for col in test_data.columns:
    test_data[col] = encoders[col].transform(test_data[col])  # 此处不能用fit_transform，因为已经fit过了
print(test_data)

test_x = test_data.iloc[:, :-1]
test_y = test_data.iloc[:, -1]

print("最优参数：", model.best_params_)
print("最优得分：", model.best_score_)
best_model = model.best_estimator_

# 预测
pred_test_y = best_model.predict(test_x)
print("真实类别：", encoders[6].inverse_transform(test_y))
print("预测类别：", encoders[6].inverse_transform(pred_test_y))
print("预测类别概率(置信概率)", best_model.predict_proba(test_x))

"""
最优参数： {'max_depth': 14, 'n_estimators': 450}
最优得分： 0.792824074074074
真实类别： ['unacc' 'acc' 'good' 'vgood']
预测类别： ['unacc' 'acc' 'good' 'good']
预测类别概率(置信概率) [[0.02444444 0.         0.97555556 0.        ]
 [0.90666667 0.         0.09333333 0.        ]
 [0.11111111 0.81555556 0.07333333 0.        ]
 [0.11222222 0.77444444 0.00666667 0.10666667]]
"""
