"""
基本流程
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 一、准备数据
indices_permutation = np.random.permutation(len(inputs))
shuffled_inputs = inputs[indices_permutation]
shuffled_targets = targets[indices_permutation]

num_validation_samples = int(0.3 * len(inputs))
val_inputs = shuffled_inputs[:num_validation_samples]
val_targets = shuffled_targets[:num_validation_samples]
training_inputs = shuffled_inputs[num_validation_samples:]
training_targets = shuffled_targets[num_validation_samples:]

# 二、定义模型
model = keras.Sequential([tf.keras.layers.Dense(1)])

# 三、编译模型
# 方式1
model.compile(optimizer="rmsprop",
              loss="mean_squared_error",
              metrics=["accuracy"])

# 方式2: 方便定义配置
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])

# 四、训练模型
model.fit(
    training_inputs,
    training_targets,
    epochs=5,
    batch_size=16,
    validation_data=(val_inputs, val_targets)
)

# 五、评估
test_metrics = model.evaluate(test_inputs, test_labels)

# 六、推理
predictions = model.predict(val_inputs, batch_size=128)
print(predictions[:10])
