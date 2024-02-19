"""
变量的使用:用于模型参数w，b
使用变量必须进行初始化
"""
import tensorflow as tf

init_w = tf.random.normal(shape=(3, 4))  # 权重初始值通常为正态分布的随机数
init_b = tf.zeros(shape=(4,))  # 偏置初始值通常为0或1

w = tf.Variable(initial_value=init_w)  # 定义权重w
b = tf.Variable(initial_value=init_b)  # 定义偏置b

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())  # 初始化变量
    w, b = sess.run([w, b])
    print(w)
    """
    [[ 0.60722107 -0.18668914  0.8541105  -1.4878786 ]
 [ 1.6078689   0.55736494 -0.65720683  0.86629814]
 [ 0.3932594  -1.4065859   1.4673705  -0.45471483]]
    """
    print(b)  # [0. 0. 0. 0.]
