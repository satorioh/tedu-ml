import tensorflow as tf

x = tf.constant(100.0, name='x')
y = tf.constant(200.0, name='y')
add = tf.add(x, y, name='add')

# 创建一个summary writer
writer = tf.summary.create_file_writer('../tf-logs/')

# 在writer的上下文管理器中记录数据
with writer.as_default():
    tf.summary.scalar('x', x, step=1)
    tf.summary.scalar('y', y, step=1)
    tf.summary.scalar('add', add, step=1)

writer.flush()
