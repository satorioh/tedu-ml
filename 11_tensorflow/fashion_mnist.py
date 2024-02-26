'''
服饰识别
模型：卷积神经网络CNN
'''
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


class FashionMnist:
    out_feature1 = 12  # 第一组的卷积核数量
    out_feature2 = 24  # 第二组的卷积核数量
    con_neurons = 512  # 全连接层神经元的数量

    def __init__(self, path):
        self.data = read_data_sets(path, one_hot=True)
        self.sess = tf.Session()

    def close(self):
        self.sess.close()

    # 初始化权重
    def init_weight_var(self, shape):
        # 截尾正态分布
        init_w = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init_w)

    # 初始化偏置
    def init_bias_var(self, shape):
        init_b = tf.constant(1.0, shape=shape)
        return tf.Variable(init_b)

    # 二维卷积
    def conv2d(self, x, w):
        return tf.nn.conv2d(x,  # 输入数据
                            w,  # 卷积核
                            strides=[1, 1, 1, 1],  # 步长 NHWC
                            padding='SAME')

    # 池化
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x,  # 输入数据
                              ksize=[1, 2, 2, 1],  # 池化区域
                              strides=[1, 2, 2, 1],  # 池化步长
                              padding='SAME')  # 填充

    # 卷积池化组
    def create_conv_pool_layer(self, input, input_c, out_c):
        '''
        卷积池化组  5*5卷积
        :param input: 输入数据
        :param input_c: 输入通道数
        :param out_c: 输出通道数
        :return:
        '''

        # 卷积核
        filter_w = self.init_weight_var([5, 5, input_c, out_c])
        # 卷积核的偏置
        b_conv = self.init_bias_var([out_c])
        # 执行卷积,激活
        h_conv = tf.nn.relu(self.conv2d(input, filter_w) + b_conv)
        # 执行池化
        h_pool = self.max_pool_2x2(h_conv)
        return h_pool

    # 全连接层
    def create_fc_layer(self, h_pool_flat, input_feature, con_neurons):
        '''
        全连接层
        :param h_pool_flat: 输入数据（一维特征）
        :param input_feature: 输入特征数量
        :param con_nrerons: 神经元数量
        :return:
        '''
        w_fc = self.init_weight_var([input_feature, con_neurons])
        b_fc = self.init_bias_var([con_neurons])

        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, w_fc) + b_fc)
        return h_fc1

    def build(self):
        # 组建CNN

        # 样本数据的占位符
        self.x = tf.placeholder('float32', [None, 784])
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        self.y = tf.placeholder('float32', [None, 10])

        # 第一组卷积池化
        h_pool1 = self.create_conv_pool_layer(x_image, 1, self.out_feature1)

        # 第二组卷积池化
        h_pool2 = self.create_conv_pool_layer(h_pool1, self.out_feature1, self.out_feature2)

        # 全连接层

        h_pool2_flat_feature = 7 * 7 * self.out_feature2
        h_pool2_flat = tf.reshape(h_pool2, [-1, h_pool2_flat_feature])

        h_fc = self.create_fc_layer(h_pool2_flat,
                                    h_pool2_flat_feature,
                                    self.con_neurons)

        # dropout层
        h_fc_drop = tf.nn.dropout(h_fc, 0.5)

        # 输出层
        w_fc = self.init_weight_var([self.con_neurons, 10])  # (512,10)
        b_fc = self.init_bias_var([10])

        pred_y = tf.matmul(h_fc_drop, w_fc) + b_fc

        # 损失函数
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=pred_y)
        # 求均值
        cross_entropy = tf.reduce_mean(loss)

        # 梯度下降:自适应梯度下降优化器
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

        # 精度
        corr = tf.equal(tf.argmax(self.y, 1),
                        tf.argmax(pred_y, 1))

        self.accuracy = tf.reduce_mean(tf.cast(corr, 'float32'))

    def train(self):
        # 初始化
        self.sess.run(tf.global_variables_initializer())
        # 批次大小
        batch_size = 100
        print('开始训练......')

        for i in range(10):  # 轮数
            total_batch = int(self.data.train.num_examples / batch_size)
            # 内层控制批次数
            total_acc = 0.0
            for j in range(total_batch):
                # 获取训练集中一个批次的数据
                train_x, train_y = self.data.train.next_batch(batch_size)

                params = {self.x: train_x, self.y: train_y}

                t, acc = self.sess.run([self.train_op, self.accuracy], feed_dict=params)
                # 平均精度
                total_acc += acc
            avg_acc = total_acc / total_batch
            print('轮数:{},精度:{}'.format(i + 1, avg_acc))

    # 评估
    def metrics(self):
        # 拿到测试集的数据
        test_x, test_y = self.data.test.next_batch(10000)
        params = {self.x: test_x, self.y: test_y}
        test_acc = self.sess.run(self.accuracy,
                                 feed_dict=params)
        print('测试集精度:', test_acc)


if __name__ == '__main__':
    mnist = FashionMnist('../fashion_mnist/')
    mnist.build()  # 搭建网络
    mnist.train()  # 训练
    mnist.metrics()  # 评估
    mnist.close()
