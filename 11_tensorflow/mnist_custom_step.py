import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Mean, CategoricalAccuracy

# 显示可用的GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load the data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data(path='mnist.npz')

# Preprocess the data, 归一化，并将样本数据从整数转换为浮点数
train_images = train_images / 255.0
test_images = test_images / 255.0

# label one-hot编码
train_labels = tf.one_hot(train_labels.astype(np.int32), depth=10)
test_labels = tf.one_hot(test_labels.astype(np.int32), depth=10)

# Model parameters
w = tf.Variable(tf.random.normal([784, 10], dtype=tf.float64))
b = tf.Variable(tf.zeros([10], dtype=tf.float64))

# Optimizer
optimizer = SGD(0.01)

# Loss and metrics
loss_object = CategoricalCrossentropy()
train_loss = Mean()
train_accuracy = CategoricalAccuracy()


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # Reshape the images
        images = tf.reshape(images, [-1, 784])
        # Build the model
        logits = tf.nn.softmax(tf.matmul(images, w) + b)
        # Compute the loss
        loss = loss_object(labels, logits)
    # Compute the gradients
    gradients = tape.gradient(loss, [w, b])
    # Update the weights
    optimizer.apply_gradients(zip(gradients, [w, b]))
    # Update the metrics
    train_loss(loss)
    train_accuracy(labels, logits)


# Create a dataset from the training images and labels
train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

# Shuffle and batch the dataset
batch_size = 32
train_data = train_data.shuffle(60000).batch(batch_size)

# Training loop
with tf.device('/GPU:0'):
    for epoch in range(100):
        for (images, labels) in train_data:
            train_step(images, labels)
        print('Epoch:', epoch, 'Mean Loss:', train_loss.result().numpy(), 'Accuracy:', train_accuracy.result().numpy())
