import pandas as pd
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt

df_ads = pd.read_csv("../data_test/advertising.csv")

# 准备输入和输出
x = df_ads.iloc[:, 0:1]
y = df_ads.sales

# 画图
# plt.scatter(x.values, y)
# plt.show()

# 拆分训练集和测试集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=0)

# 归一化
scaler_x = preprocessing.MinMaxScaler().fit(x_train.values)
scaler_y = preprocessing.MinMaxScaler().fit(y_train.values.reshape(-1, 1))
x_train = scaler_x.transform(x_train.values)
x_test = scaler_x.transform(x_test.values)
y_train = scaler_y.transform(y_train.values.reshape(-1, 1))
y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

# 画归一化处理后的图
plt.scatter(x_test, y_test)
plt.show()

