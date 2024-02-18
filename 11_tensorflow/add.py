"""
张量相加
"""
import tensorflow as tf

# 创建两个常量张量
x = tf.constant(100.0)
y = tf.constant(200.0)
temp = tf.add(x, y)

with tf.compat.v1.Session() as sess:
    print(sess.run(temp))  # 300.0
