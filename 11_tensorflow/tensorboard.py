import tensorflow as tf

x = tf.constant(100.0, name='x')
y = tf.constant(200.0, name='y')
add = tf.add(x, y, name='add')

with tf.compat.v1.Session() as session:
    print(session.run(add))
    writer = tf.summary.FileWriter('../tf-logs/', graph=session.graph)
