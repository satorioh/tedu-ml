import numpy as np
import pandas as pd

data = pd.read_json('../data_test/ratings.json')
fracture = data.loc['Fracture']
print(data)

# 一维数据求平均值
print(np.mean(fracture))  # 4.0
print(fracture.mean())  # 4.0
print("==" * 20)

# 二维数据求平均值
print(np.mean(data, axis=1))
print(data.mean(axis=1))  # 横向，即求每一行的
"""
Inception           2.800000
Pulp Fiction        3.714286
Anger Management    2.375000
Fracture            4.000000
Serendipity         2.500000
Jerry Maguire       3.416667
"""
