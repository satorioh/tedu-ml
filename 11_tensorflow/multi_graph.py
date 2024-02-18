"""
多个图
"""
import tensorflow as tf

# 创建两个常量张量
x = tf.constant(100.0)
y = tf.constant(200.0)
temp = tf.add(x, y)

# 获取默认的图
graph = tf.compat.v1.get_default_graph()
print('默认的图', graph)  # <tensorflow.python.framework.ops.Graph object at 0x7f92c87e3f40>

# 创建一个新的图
new_graph = tf.Graph()
print('新的图', new_graph)  # <tensorflow.python.framework.ops.Graph object at 0x7fa8f45b2520>

# 临时将新建的图设为默认的图
with new_graph.as_default():
    new_op = tf.constant('hello kitty')

# 运行新建的图中的op
with tf.compat.v1.Session(graph=new_graph) as sess:
    print(sess.run(new_op))  # b'hello kitty'

# 运行默认的图
with tf.compat.v1.Session() as sess:
    print(sess.run(temp))  # 300.0
