"""
银行客户流失分类问题：
标签分类不平衡
全连接网络
"""
import numpy as np  # 导入NumPy数学工具箱
import pandas as pd  # 导入Pandas数据处理工具箱
import matplotlib.pyplot as plt  # 导入matplotlib画图工具箱
import seaborn as sns  # 导入seaborn画图工具箱
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler  # 导入特征缩放器

df_bank = pd.read_csv("../data_test/BankCustomer.csv")  # 读取文件
df_bank.head()  # 显示文件前5行

# 显示不同特征的分布情况
features = ['City', 'Gender', 'Age', 'Tenure',
            'ProductsNo', 'HasCard', 'ActiveMember', 'Exited']
fig = plt.subplots(figsize=(15, 15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i + 1)
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(x=j, data=df_bank)
    plt.title("No. of costumers")

# --------------------------数据预处理--------------------------#
# 把二元类别文本数字化
df_bank['Gender'].replace("Female", 0, inplace=True)
df_bank['Gender'].replace("Male", 1, inplace=True)
print("Gender unique values", df_bank['Gender'].unique())  # Gender unique values [0 1]

# 把多元类别转换成多个二元哑变量，然后贴回原始数据集
d_city = pd.get_dummies(df_bank['City'], prefix="City")
df_bank = [df_bank, d_city]
df_bank = pd.concat(df_bank, axis=1)

y = df_bank['Exited']
X = df_bank.drop(['Name', 'Exited', 'City'], axis=1)
X.head()

# --------------------------拆分数据集--------------------------#
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=0)

# --------------------------特征缩放--------------------------#
sc = StandardScaler()  # 特征缩放器
X_train = sc.fit_transform(X_train)  # 拟合并应用于训练集
X_test = sc.transform(X_test)  # 训练集结果应用于测试集

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=12, activation='relu'),  # 输入层
    tf.keras.layers.Dense(units=24, activation='relu'),  # 隐层
    tf.keras.layers.Dense(units=1, activation='sigmoid')  # 输出层，二分类
])

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train,  # 指定训练集
                    epochs=100,  # 指定训练的轮次
                    batch_size=64,  # 指定数据批量
                    validation_data=(X_test, y_test))  # 指定验证集,这里为了简化模型，直接用测试集数据进行验证


# 绘制训练过程中的损失和准确率
# 这段代码参考《Python深度学习》一书中的学习曲线的实现
def show_history(history):  # 显示训练过程中的学习曲线
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


show_history(history)  # 调用这个函数，并将神经网络训练历史数据作为参数输入
