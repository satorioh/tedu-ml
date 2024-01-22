"""
掩码操作: 求100以内能同时被3和7整除的数字
"""
import numpy as np

# 布尔掩码
a = np.arange(1, 101)
b = np.arange(1, 101)
# mask = b % 21 == 0
mask = (b % 3 == 0) & (b % 7 == 0)
print(a[mask])  # [21 42 63 84]

# 索引掩码
cars = np.array(['BMW', 'Audi', 'Tesla', 'QQ'])
mask = [3, 2, 1, 0]
mask2 = [2, 1]
mask3 = [3, 3, 2, 2]
print(cars[mask])  # ['QQ' 'Tesla' 'Audi' 'BMW']
print(cars[mask2])  # ['Tesla' 'Audi']
print(cars[mask3])  # ['QQ' 'QQ' 'Tesla' 'Tesla']
