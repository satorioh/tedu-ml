import tensorflow as tf

# 定义
hello = tf.constant('Hello, TensorFlow!')  # 定义一个常量
print(hello)  # Tensor("Const:0", shape=(), dtype=string)

# 运行
with tf.compat.v1.Session() as sess:
    print(sess.run(hello))  # b'Hello, TensorFlow!'
