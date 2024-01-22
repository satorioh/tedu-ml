"""
测试时间日期类型
"""
import numpy as np

array01 = np.array(['2020', '2021-01-01', '2022-01-01 08:08:08'])
print(array01)
print(array01.dtype)  # <U19

# str --> datetime64
array02 = array01.astype("datetime64")
print(array02)  # 以精度最高的元素为准['2020-01-01T00:00:00' '2021-01-01T00:00:00' '2022-01-01T08:08:08']

array03 = array01.astype("datetime64[D]")  # 精确到Y/M/D/h/m/s
print(array03)  # 以转换时设置的精度为准['2020-01-01T00' '2021-01-01T00' '2022-01-01T08']

# datetime64 --> int64
array04 = array03.astype("int64")
print(array04)  # [18262 18628 18993]，即距1970/1/1 00:00:00过去了多少[Y/M/D/h/m/s]
