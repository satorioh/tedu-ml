"""
张量的形状改变
"""
import tensorflow as tf

x = tf.constant([[1, 2, 3], [4, 5, 6]])

print(x.shape)  # (2, 3)

x_new = tf.reshape(x, [1, 2, 3])
print(x_new.shape)  # (1, 2, 3)
