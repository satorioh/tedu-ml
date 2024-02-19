"""
占位符：一般用于样本数据的占位
"""
import numpy as np
import tensorflow as tf

x = tf.compat.v1.placeholder(tf.float32, shape=(None, 3))

with tf.compat.v1.Session() as sess:
    x_data = np.arange(1, 7).reshape(2, 3)
    result = sess.run(x, feed_dict={x: x_data})
    print(result)  # [[1. 2. 3.][4. 5. 6.]]
