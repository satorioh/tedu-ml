"""
张量相加
"""
import tensorflow as tf

# 创建两个常量张量
x = tf.constant(100.0)
y = tf.constant(200.0)
temp = tf.add(x, y)

print(temp.numpy())  # 300.0
