"""
波士顿房价预测: 使用线性模型预测
506个样本，13列特征+1个价格中位数
"""
import sklearn.model_selection as model_selection
import sklearn.linear_model as linear_model
import sklearn.preprocessing as preprocessing
import sklearn.pipeline as pipeline
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

raw_df = pd.read_csv("../../data_test/boston.txt", sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
print(data.shape)
print(target.shape)


def get_feature_names(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    start_index = lines.index(' Variables in order:\n') + 1
    end_index = start_index + 14
    features = [line.strip().split()[0] for line in lines[start_index:end_index] if line.strip()]
    return features


# features = get_feature_names('../../data_test/boston.txt')
# print(features)

x = data
y = target
# 划分训练集和测试集（训练集数据不能有顺序）
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=7)


# 构建模型
def get_model_score(model, model_name):
    print(f"-----------{model_name}-----------")
    model.fit(x_train, y_train)
    lm_pred_y_train = model.predict(x_train)
    lm_pred_y_test = model.predict(x_test)
    print("训练集MSE：", metrics.mean_squared_error(y_train, lm_pred_y_train))
    print("测试集MSE：", metrics.mean_squared_error(y_test, lm_pred_y_test))
    print("训练集R2：", metrics.r2_score(y_train, lm_pred_y_train))
    print("测试集R2：", metrics.r2_score(y_test, lm_pred_y_test))


model_dict = {
    "LinearRegression": linear_model.LinearRegression(),
    "Ridge": linear_model.Ridge(),
    "PolyRegression": pipeline.make_pipeline(preprocessing.PolynomialFeatures(2), linear_model.LinearRegression()),
    "Poly+Ridge": pipeline.make_pipeline(preprocessing.PolynomialFeatures(2), linear_model.Ridge(alpha=100)),
}

for model_name, model in model_dict.items():
    get_model_score(model, model_name)

"""
-----------LinearRegression-----------
训练集MSE： 19.638717311375427
测试集MSE： 34.056481348874556
训练集R2： 0.7698532963729757
测试集R2： 0.5785415472763416
-----------Ridge-----------
训练集MSE： 19.78037655220107
测试集MSE： 34.717269008935574
训练集R2： 0.7681931875788315
测试集R2： 0.5703641157344477
-----------PolyRegression-----------
训练集MSE： 5.663956205899134
测试集MSE： 30.948647744617492
训练集R2： 0.9336239312574424
测试集R2： 0.6170018546919829
-----------Poly+Ridge-----------
训练集MSE： 6.475034242532565
测试集MSE： 20.620495723334447
训练集R2： 0.924118883979872
测试集R2： 0.7448156157729856
"""
