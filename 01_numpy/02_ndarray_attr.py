import numpy as np

array01 = np.zeros(10)
array02 = np.zeros(shape=(2, 3))
print(array01.shape)  # (10,)
print(array02.shape)  # (2, 3)
print(array01.ndim)  # 1
print(array02.ndim)  # 2
print("==" * 20)

array03 = np.arange(1, 11)
print(array03.dtype)  # int64
array04 = array03.astype('float64')
print(array04)  # [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
print("==" * 20)

ary = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8]
])
print(ary.size)  # 8
print(len(ary))  # 2
print("==" * 20)

a = np.array([[[1, 2],
               [3, 4]],
              [[5, 6],
               [7, 8]]])
print(a, a.shape)
print(a[0])
print(a[0][0], a[0, 0])  # [1 2] [1 2]
print(a[0][0][0], a[0, 0, 0])  # 1 1

b = np.array(["a", "abc", "dfe"])
print(b.dtype)
print("==" * 20)

c = np.arange(1, 11).reshape(2, 5)
print(c.itemsize)  # 8
print(c.nbytes)  # 80
print(c)  # [[ 1  2  3  4  5][ 6  7  8  9 10]]
print(c.T)  # [[ 1  6][ 2  7][ 3  8][ 4  9][ 5 10]]
