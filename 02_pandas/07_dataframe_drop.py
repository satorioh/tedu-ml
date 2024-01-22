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

# 删除行
df.drop(['c', 'd'], inplace=True)
print(df)
"""
   one  two  three
a  1.0    1    1.0
b  2.0    2    NaN
"""

# 删除列
df.drop(columns=['three', 'two'], inplace=True)
print(df)
"""
   one
a  1.0
b  2.0
"""
