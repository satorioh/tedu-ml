"""
DataFrame列操作
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

# 列标签访问
print(df['one'])
"""
a    1.0
b    2.0
c    3.0
d    NaN
"""

# 掩码访问
print(df[['one', 'two']])
"""
   one  two
a  1.0    1
b  2.0    2
c  3.0    3
d  NaN    4
"""

# 掩码访问方式2
print(df[df.columns[:-1]])  # 结果同上

print("==" * 20)

# 列添加
df['four'] = pd.Series([90, 80, 70, 60], index=['a', 'b', 'c', 'd'])
print(df)
"""
   one  two  three  four
a  1.0    1    1.0    90
b  2.0    2    NaN    80
c  3.0    3    3.0    70
d  NaN    4    4.0    60
"""
print("==" * 20)

# 列删除
# 删除一列
del df['four']
df.pop('two')

# 删除多列
df.drop(['four', 'two'], axis=1, inplace=True)
print(df)
