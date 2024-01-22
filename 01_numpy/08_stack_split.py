"""
数组的组合和拆分
"""
import numpy as np

x = np.arange(1, 7).reshape(2, 3)
y = np.arange(7, 13).reshape(2, 3)

# 垂直
a = np.vstack((x, y))
print(a)  # [[ 1  2  3][ 4  5  6][ 7  8  9][10 11 12]]
b1, b2 = np.vsplit(a, 2)
print(b1)  # [[1 2 3][4 5 6]]
print(b2)  # [[ 7  8  9][10 11 12]]
print("==" * 20)

# 水平
c = np.hstack((x, y))
print(c)  # [[ 1  2  3  7  8  9][ 4  5  6 10 11 12]]
c1, c2 = np.hsplit(c, 2)
print(c1)  # [[1 2 3][4 5 6]]
print(c2)  # [[ 7  8  9][10 11 12]]
print("==" * 20)

# 深度
d = np.dstack((x, y))
print(d)  # [[[ 1  7][ 2  8][ 3  9]][[ 4 10][ 5 11][ 6 12]]]
d1, d2 = np.dsplit(d, 2)
print(d1)  # [[[1][2][3]] [[4][5][6]]]
print(d2)  # [[[7][8][9]] [[10][11][12]]]
print("==" * 20)
