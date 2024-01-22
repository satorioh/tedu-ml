import numpy as np

list01 = [1, 2, 3, 4, 5]
array = np.array(list01)
print(array)
print(list01)
print("==" * 20)

a = np.arange(0, 5, 1)
print(a)
b = np.arange(0, 10, 2)
print(b)
c = np.arange(0.1, 1.1, 0.2)
print(c)
print("==" * 20)

d = np.zeros(10)
print(d)  # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
e = np.zeros(10, dtype='int32')
print(e)  # [0 0 0 0 0 0 0 0 0 0]
f = np.zeros(shape=(3, 2))
print(f)  # [[0. 0.][0. 0.][0. 0.]]
print("==" * 20)

g = np.ones(shape=(3, 2), dtype='int32')
print(g)  # [[1 1][1 1][1 1]]
print("==" * 20)

list02 = [[1, 2, 3], [4, 5, 6]]
array02 = np.array(list02)
print(array02)  # [[1 2 3][4 5 6]]
zero_like_array = np.zeros_like(array02)
one_like_array = np.ones_like(array02)
print(zero_like_array)  # [[0 0 0][0 0 0]]
print(one_like_array)  # [[1 1 1][1 1 1]]

# 生成值都为0.2的数组
h = np.zeros(10) + 0.2
print(h)

# 线性拆分
pai = np.linspace(-3.14, 3.14, 100)
print(pai)
