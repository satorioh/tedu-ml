"""
随机森林分类树：根据特征预测小汽车等级
"""
import pandas as pd
import sklearn.preprocessing as preprocessing
import sklearn.ensemble as ensemble

data = pd.read_csv('../data_test/car.txt', header=None)
# print(data[0].unique())

encoders = {}
train_data = pd.DataFrame()

for col in data.columns:
    encoder = preprocessing.LabelEncoder()
    encoders[col] = encoder
    train_data[col] = encoder.fit_transform(data[col])
print(train_data)

# 整理输入和输出
train_x = train_data.iloc[:, :-1]
train_y = train_data.iloc[:, -1]

# 构建随机森林
model = ensemble.RandomForestClassifier(max_depth=6, n_estimators=400, random_state=7)
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

# 预测
pred_test_y = model.predict(test_x)
print("真实类别：", encoders[6].inverse_transform(test_y))
print("预测类别：", encoders[6].inverse_transform(pred_test_y))
