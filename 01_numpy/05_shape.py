"""
变形操作
"""
import numpy as np

a = np.arange(1, 10)
print(a)  # [ 1  2  3  4  5  6  7  8  9 10]

b = a.reshape(3, 3)
print(a)  # a不变[ 1  2  3  4  5  6  7  8  9 10]
print(b)  # [[1 2 3][4 5 6][7 8 9]]

# 修改a中的元素，b也变了
a[0] = 666
print(b)  # [[666   2   3][  4   5   6][  7   8   9]]
print("==" * 20)

c = b.ravel()
print(c)  # [666   2   3   4   5   6   7   8   9]
print(b)  # [[666   2   3][  4   5   6][  7   8   9]]
print("==" * 20)

d = np.arange(1, 9).reshape(2, 2, 2)
print(d)  # [[[1 2][3 4]][[5 6][7 8]]]
e = d.flatten()
print(e)  # [1 2 3 4 5 6 7 8]
d[0, 0, 0] = 999
print(e)  # e不变 [1 2 3 4 5 6 7 8]
print("==" * 20)

f = np.arange(1, 9)
f.resize(2, 2, 2)
print(f)  # [[[1 2][3 4]][[5 6][7 8]]]
