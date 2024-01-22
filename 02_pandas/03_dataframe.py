import pandas as pd

# 一维列表转df
list01 = [1, 2, 3, 4, 5]
df = pd.DataFrame(list01)
print(df.shape)  # (5, 1)
print(df)
"""
   0
0  1
1  2
2  3
3  4
4  5
"""

# 二维列表转df
list02 = [['Alex', 10], ['Bob', 12], ['Clarke', 13]]
df2 = pd.DataFrame(list02)
print(df2.shape)  # (3, 2)
print(df2)
"""
        0   1
0    Alex  10
1     Bob  12
2  Clarke  13
"""
print("==" * 20)

# 设置行列索引
list03 = [['Alex', 10], ['Bob', 12], ['Clarke', 13]]
df3 = pd.DataFrame(list03, index=['r1', 'r2', 'r3'], columns=['Name', 'age'])
print(df3)
"""
      Name  age
r1    Alex   10
r2     Bob   12
r3  Clarke   13
"""
print("==" * 20)

# 从字典来创建DataFrame
dict01 = {'Name': ['Tom', 'Jack', 'Steve', 'Ricky'], 'Age': [28, 34, 29, 42]}
df4 = pd.DataFrame(dict01, index=['r1', 'r2', 'r3', 'r4'])
print(df4)
"""
     Name  Age
r1    Tom   28
r2   Jack   34
r3  Steve   29
r4  Ricky   42
"""

# 自动为缺失数据设置NaN
dict02 = {'Name': pd.Series(['Tom', 'Jack', 'Steve', 'Ricky'], index=['r1', 'r2', 'r3', 'r4']),
          'Age': pd.Series([28, 34, 29], index=['r1', 'r2', 'r4'])}
df5 = pd.DataFrame(dict02)
print(df5)
"""
 Name   Age
r1    Tom  28.0
r2   Jack  34.0
r3  Steve   NaN
r4  Ricky  29.0
"""
