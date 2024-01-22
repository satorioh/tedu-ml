import pandas as pd
import numpy as np

list01 = [100, 98, 33, 20]
s1 = pd.Series(list01)
print(s1)
"""
0    100
1    abc
2     33
3     20
"""
s2 = pd.Series(list01, index=['zs', 'ls', 'ww', 'zl'])
print(s2)
"""
zs    100
ls     98
ww     33
zl     20
"""

# 从字典创建一个Series
data = {'100': '张三', '101': '李四', '102': '王五'}
s3 = pd.Series(data)
print(s3)

# 从标量创建一个Series
s4 = pd.Series(5, index=range(5))
print(s4)
"""
0    5
1    5
2    5
3    5
4    5
"""
