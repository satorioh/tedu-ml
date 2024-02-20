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


# 3. 构建损失函数(mse)
def compute_loss():
    pred_y = tf.matmul(x, weight) + bias
    return tf.reduce_mean(tf.square(y - pred_y))


# 4. 梯度下降优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# 定义收集损失函数
summary_writer = tf.summary.create_file_writer('./logs')

# 5. 训练模型
for i in range(500):
    with tf.GradientTape() as tape:
        loss = compute_loss()
    grads = tape.gradient(loss, [weight, bias])
    optimizer.apply_gradients(zip(grads, [weight, bias]))
    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=i)
    print(f"step: {i}, weight: {weight.numpy()}, bias: {bias.numpy()}, loss: {loss.numpy()}")

summary_writer.flush()
