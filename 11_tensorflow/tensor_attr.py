"""
张量的属性
"""
import tensorflow as tf

# 创建一个张量
x = tf.constant(100.0)


print('name', x.name)  # name Const:0
print('dtype', x.dtype)  # <dtype: 'float32'>
print('shape', x.shape)  # ()
print(('graph', x.graph))  # <tensorflow.python.framework.ops.Graph object at 0x7f3e3c3e3f60>
