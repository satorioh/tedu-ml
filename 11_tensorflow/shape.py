"""
张量的形状改变
"""
import tensorflow as tf

x = tf.compat.v1.placeholder(tf.float32, shape=(None, 3))

# 静态形状
print(x.get_shape())  # (?, 3)

x.set_shape([4, 3])
print(x.get_shape())  # (4, 3)
x.set_shape([3, 3])  # 报错，静态形状一旦设置不能再次设置

# 动态形状
x_new = tf.reshape(x, [1, 3, 4])
print(x_new.get_shape())  # (1, 3, 4)
