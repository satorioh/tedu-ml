import pandas as pd
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
from sklearn.compose import ColumnTransformer

data = pd.read_csv("../data_test/heart.csv")

# 整理输入输出
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 拆分训练集和测试集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

# 创建OneHotEncoder实例
encoder = preprocessing.OneHotEncoder(sparse_output=False)
# 定义需要进行独热编码的列
columns_to_encode = ['cp', 'slope', 'thal']

# 创建ColumnTransformer实例，用于对指定列进行转换
column_transformer = ColumnTransformer(
    transformers=[
        ('encoder', encoder, columns_to_encode)
    ],
    remainder='passthrough'  # 保留未指定的列
)

# 对训练集和测试集进行转换
x_train_encoded = column_transformer.fit_transform(x_train)
x_test_encoded = column_transformer.transform(x_test)

# 构建逻辑回归模型
model = linear_model.LogisticRegression(solver='liblinear')
# 训练模型
model.fit(x_train_encoded, y_train)
# 预测
y_pred = model.predict(x_test_encoded)
# 评估
print('真实结果：', y_test.values)
print('预测结果：', y_pred)
print('准确率：', metrics.accuracy_score(y_test, y_pred))  # 0.9016393442622951
