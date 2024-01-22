"""
读取SaLary_data.csv中的数据
以YearsExperience 为x，Salary作为y画散点图
颜色随着Salary的变化而变化
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("../data_test/Salary_Data.csv")
print(data)
print("==" * 20)
x = data['YearsExperience']
y = data['Salary']
plt.scatter(x, y, c=y, cmap='jet')
plt.show()
