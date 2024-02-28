"""
PassengerId：用户id
survival：是否生还，0-否，1-是
pclass：舱位，1-头等舱，2-二等，3-三等
name：姓名
sex：性别
Age：年龄
sibsp：在船上的兄弟/配偶数
parch：在船上父母/孩子数
ticket：票号
fare：票价
cabin：Cabin number；客舱号
embarked：登船地点
"""
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

train_data = pd.read_csv("../../data_test/titanic/train.csv")
test_data = pd.read_csv("../../data_test/titanic/test.csv")
full_data = pd.concat([train_data, test_data], ignore_index=True)

# 1.填充缺失值
# Embarked登船地点
# print(full_data[full_data['Embarked'].isnull()])  # 查看缺失值对应的数据
full_data['Embarked'] = full_data.Embarked.fillna('C')  # Pclass=1，Fare=80，Embarked=C
# Age年龄
full_data['Age'] = full_data.Age.fillna(full_data.Age.mean())  # 用平均值填充
# Fare票价
full_data['Fare'] = full_data.Fare.fillna(full_data.Fare.mean())  # 用平均值填充

# 2.生成新特征
# 生成Title特征
full_data['Title'] = full_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Dona": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
}
full_data['Title'] = full_data['Title'].map(Title_Dictionary)
# print(full_data['Title'].value_counts())

# 生成FamilySize特征
full_data['familyNum'] = full_data['Parch'] + full_data['SibSp'] + 1


# 我们按照家庭成员人数多少，将家庭规模分为“小、中、大”三类：
def family_size(family_num):
    if family_num == 1:
        return 0
    elif (family_num >= 2) & (family_num <= 4):
        return 1
    else:
        return 2


full_data['familySize'] = full_data['familyNum'].map(family_size)
# print(full_data['familySize'].value_counts())

# 取Cabin首字符作为相关特征
full_data['Cabin'] = full_data.Cabin.fillna('U')
full_data['Cabin'] = full_data['Cabin'].map(lambda c: c[0])

# full_data.to_csv("../data_test/titanic/prepare_data.csv")

# 3.删除无用特征
test_new_data_with_id = full_data.iloc[891:, :].copy()
full_data = full_data.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'familyNum'], axis=1)

# 划分训练集和测试集
train_new_data = full_data.iloc[:891, :]
test_new_data = full_data.iloc[891:, :]
train_x = train_new_data.drop(['Survived'], axis=1)
train_y = train_new_data['Survived']
test_x = test_new_data.drop(['Survived'], axis=1)

# 进行one-hot编码
# 创建OneHotEncoder实例
encoder = preprocessing.OneHotEncoder(sparse_output=False)
# 定义需要进行独热编码的列
columns_to_encode = ['Sex', 'Embarked', 'Cabin', 'Title']

# 创建ColumnTransformer实例，用于对指定列进行转换
column_transformer = ColumnTransformer(
    transformers=[
        ('encoder', encoder, columns_to_encode)
    ],
    remainder='passthrough'  # 保留未指定的列
)

train_x_encoded = column_transformer.fit_transform(train_x)
test_x_encoded = column_transformer.transform(test_x)

# 4.模型训练
# 设置kfold，交叉采样法拆分数据集
kfold = StratifiedKFold(n_splits=10)

# 汇总不同模型算法
classifiers = [SVC(), DecisionTreeClassifier(), RandomForestClassifier(), ExtraTreesClassifier(),
               GradientBoostingClassifier(), KNeighborsClassifier(), LogisticRegression(), LinearDiscriminantAnalysis()]

# 不同机器学习交叉验证结果汇总
cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, train_x_encoded, train_y,
                                      scoring='accuracy', cv=kfold, n_jobs=-1))
# 求出模型得分的均值和标准差
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

# 汇总数据
cvResDf = pd.DataFrame({'cv_mean': cv_means,
                        'cv_std': cv_std,
                        'algorithm': ['SVC', 'DecisionTreeCla', 'RandomForestCla', 'ExtraTreesCla',
                                      'GradientBoostingCla', 'KNN', 'LR', 'LinearDiscrimiAna']})
print(cvResDf)

# GradientBoostingClassifier模型
# model_GBC = GradientBoostingClassifier(learning_rate=0.1, loss='log_loss', max_depth=4, max_features=0.3,
#                                        min_samples_leaf=100, n_estimators=300)
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss': ['exponential', 'log_loss'],
                 'n_estimators': [100, 200, 300],
                 'learning_rate': [0.1, 0.05, 0.01],
                 'max_depth': [4, 8],
                 'min_samples_leaf': [100, 150],
                 'max_features': [0.3, 0.1]
                 }
model_GBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=kfold,
                         scoring="accuracy")
model_GBC.fit(train_x_encoded, train_y)
print(f"GradientBoostingClassifier模型得分：{model_GBC.best_score_}")  # 0.8316978776529338
print(
    f"GradientBoostingClassifier最优参数：{model_GBC.best_params_}")  # {'learning_rate': 0.1, 'loss': 'log_loss', 'max_depth': 4, 'max_features': 0.3, 'min_samples_leaf': 100, 'n_estimators': 300}

# GradientBoostingClassifier模型得分：0.826067415730337
# GradientBoostingClassifier最优参数：{'learning_rate': 0.1, 'loss': 'log_loss', 'max_depth': 8, 'max_features': 0.3, 'min_samples_leaf': 100, 'n_estimators': 300}

best_model = model_GBC.best_estimator_
pred_test_y = model_GBC.predict(test_x_encoded)

output = pd.DataFrame(
    {"PassengerId": test_new_data_with_id["PassengerId"], "Survived": pred_test_y.astype("int64")}
)
output.to_csv("../data_test/titanic/my_submission.csv", index=False)
print("Your submission was successfully saved!")

# params_grid = {
#     'max_depth': np.arange(4, 18),
#     'n_estimators': np.arange(100, 801, 50)
# }
# RF = ensemble.RandomForestClassifier(random_state=42)
# model_RF = GridSearchCV(RF, param_grid=params_grid, cv=3,
#                         scoring="accuracy")
# model_RF.fit(train_x_encoded, train_y)
# print(f"RandomForestClassifier模型得分：{model_RF.best_score_}") # 0.8305274971941637
