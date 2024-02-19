"""
创建张量的方式
"""
import tensorflow as tf
import numpy as np

tensor_1d = tf.constant([1, 2, 3, 4, 5])
tensor_2d = tf.constant(np.arange(1, 7).reshape(2, 3))
tensor = tf.constant(100.0, shape=[2, 3])  # 创建一个2x3的张量，元素全是100.0

tensor_normal_distribution = tf.random.normal([2, 3])  # 创建一个2x3的张量，元素服从正态分布
tensor_zeros = tf.zeros([2, 3])  # 创建一个2x3的张量，元素全是0
tensor_ones = tf.ones([2, 3])  # 创建一个2x3的张量，元素全是1
tensor_zeros_like = tf.zeros_like(tensor_1d)  # 创建一个和tensor_1d形状一样的张量，元素全是0
