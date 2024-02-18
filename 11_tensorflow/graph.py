"""
图
"""
import tensorflow as tf

# 创建两个常量张量
x = tf.constant(100.0)
y = tf.constant(200.0)
temp = tf.add(x, y)

# 获取默认的图
graph = tf.compat.v1.get_default_graph()
print('默认的图', graph)  # <tensorflow.python.framework.ops.Graph object at 0x7f92c87e3f40>

with tf.compat.v1.Session() as sess:
    print(sess.run(temp))  # 300.0
    print(x.graph)  # <tensorflow.python.framework.ops.Graph object at 0x7f92c87e3f40>
    print(y.graph)  # <tensorflow.python.framework.ops.Graph object at 0x7f92c87e3f40>
    print(temp.graph)  # <tensorflow.python.framework.ops.Graph object at 0x7f92c87e3f40>
    print(sess.graph)  # <tensorflow.python.framework.ops.Graph object at 0x7f92c87e3f40>
