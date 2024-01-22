import pandas as pd

list01 = [100, 98, 33, 20]
s1 = pd.Series(list01, index=['zs', 'ls', 'ww', 'zl'])

# 索引
print(s1[-1])  # 20
print(s1['zl'])  # 20

# 切片
print(s1[:3])  # 结果是Series
print(s1['zs':'ww'])  # 同上，标签切片时包含结束值
"""
zs    100
ls     98
ww     33
"""

# 掩码
print(s1[[1, 2]])
"""
ls    98
ww    33
"""
print(s1[['ls', 'zl']])
"""
ls    98
zl    20
"""

print("==" * 20)

print(s1.values)  # [100  98  33  20]
print(s1.index)  # Index(['zs', 'ls', 'ww', 'zl'], dtype='object')
print(s1.dtype)  # int64
print(s1.size)  # 4
print(s1.ndim)  # 1
print(s1.shape)  # (4,)
