"""
掩码操作
"""
import numpy as np

# 布尔掩码
# 求100以内3的倍数
list01 = range(1, 101)
bool_list = np.arange(1, 101) % 3 == 0
print(bool_list)
a = np.array(list01)
print(a[bool_list])  # [ 3  6  9 12 15 18 21 24 27 30 33... 99]
