"""
张量类型转换
"""
import tensorflow as tf

tensor = tf.ones(shape=(10,), dtype='bool')  # [ True  True  True  True  True  True  True  True  True  True]
temp = tf.cast(tensor, dtype='float32')

with tf.compat.v1.Session() as sess:
    print(sess.run(temp))  # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
