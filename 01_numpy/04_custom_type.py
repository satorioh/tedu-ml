"""
自定义复合类型
列与列之间可以是不同的类型，但是在同一列内，类型必须相同
"""
import numpy as np

data = [('zs', [100, 100, 100], 18), ('ls', [90, 90, 90], 19), ('ww', [80, 80, 80], 20)]
# # 方式一
a = np.array(data, dtype='U2, 3int32, int32')
print(a)

# 求三个人年龄的平均值
avg_age = a['f2'].mean()
print(avg_age)
print("==" * 20)

# 方式二
b = np.array(data, dtype={
    'names': ['name', 'scores', 'age'],
    'formats': ['U2', '3int32', 'int32']})
print(b['age'].mean())
print("==" * 20)

# 方式三
c = np.array(data, dtype=[
    ('name', 'str_', 2),
    ('scores', 'int32', 3),
    ('age', 'int32', 1)])
print(c['age'].mean())
