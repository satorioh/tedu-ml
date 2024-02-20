"""
张量的数学计算
"""
import tensorflow as tf

x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
y = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)

# 加法
add = tf.add(x, y)
# 矩阵乘法
matmul = tf.matmul(x, y)
# log
log = tf.math.log(x)
# 求和
reduce_sum = tf.reduce_sum(x, axis=1)  # 按行求和

# 片段和
data = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9])
ids = tf.constant([0, 0, 0, 1, 1, 1, 2, 2, 2])
segment_sum = tf.math.segment_sum(data, ids)


print(add.numpy())  # [[2. 4.][6. 8.]]
print(matmul.numpy())  # [[ 7. 10.][15. 22.]]
print(log.numpy())  # [[0.        0.6931472][1.0986123 1.3862944]]
print(reduce_sum.numpy())  # [3. 7.]
print(segment_sum.numpy())  # [6 15 24]
