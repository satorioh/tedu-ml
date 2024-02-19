"""
线性回归
"""
import tensorflow as tf

# 1. 准备数据
x = tf.random.normal([100, 1], mean=1.75, stddev=0.5)
# 假设 y = 2x + 5
y = tf.matmul(x, [[2.0]]) + 5.0  # y_true数据

# 2. 构建线性模型
weight = tf.Variable(tf.random.normal([1, 1]))
bias = tf.Variable(0.0)
pred_y = tf.matmul(x, weight) + bias

# 3. 构建损失函数(mse)
loss = tf.reduce_mean(tf.square(y - pred_y))

# 4. 梯度下降优化器
train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
# 定义收集损失函数
tf.summary.scalar("loss", loss)
merged = tf.summary.merge_all()

# 5. 训练模型
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print(f"weight: {weight.eval()}, bias: {bias.eval()}")

    fw = tf.summary.FileWriter("../logs", graph=sess.graph)
    for i in range(500):
        sess.run(train_op)
        # 执行一次梯度下降，收集一次损失值
        summary = sess.run(merged)
        # 将损失值写入事件文件
        fw.add_summary(summary, i)
        print(f"step: {i}, weight: {weight.eval()}, bias: {bias.eval()}, loss: {loss.eval()}")
