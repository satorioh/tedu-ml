"""
DataFrame行操作
"""
import pandas as pd

d = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
     'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd']),
     'three': pd.Series([1, 3, 4], index=['a', 'c', 'd'])}

df = pd.DataFrame(d)
print(df)
"""
   one  two  three
a  1.0    1    1.0
b  2.0    2    NaN
c  3.0    3    3.0
d  NaN    4    4.0
"""
print("==" * 20)

# df.loc[行操作,列操作]
print(df.loc['a'])
"""
one      1.0
two      1.0
three    1.0
"""
print("==" * 20)

# 获取所有样本，不要最后一列，返回二维数组
print(df.iloc[:, :-1])
"""
   one  two
a  1.0    1
b  2.0    2
c  3.0    3
d  NaN    4
"""

# 获取所有记录的最后一列，返回一维数组
print(df.iloc[:, -1])
"""
a    1.0
b    NaN
c    3.0
d    4.0
"""
print("==" * 20)
