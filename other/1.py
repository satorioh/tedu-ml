import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../data_test/Salary_Data.csv')
print(data.describe())
data.hist(bins=50, figsize=(20,15))
plt.show()