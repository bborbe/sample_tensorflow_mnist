#
# https://www.tensorflow.org/tutorials/quickstart/beginner
# https://www.tensorflow.org/tutorials/keras/save_and_load
#

import os

# TensorFlow and tf.keras
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print(tf.__version__)

# checkpoint_path = "checkpoints/cp.ckpt"
checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
)

# Load a dataset
mnist = tf.keras.datasets.mnist

# x_train: uint8 NumPy array of grayscale image data with shapes
# y_train: uint8 NumPy array of digit labels (integers in range 0-9)
# x_test: uint8 NumPy array of grayscale image data with shapes
# y_test: uint8 NumPy array of digit labels (integers in range 0-9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print('load dataset completed')

# Build a machine learning model

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

print(model.summary())
print('define model completed')

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

predictions = model(x_test[:1]).numpy()
print(predictions)
print('{0:.10f}'.format(loss_fn(y_test[:1], predictions).numpy()))
print('print prediction of untrained model completed')

# Configure compiler
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
print('configure compiler completed')

latest = tf.train.latest_checkpoint(checkpoint_dir)
print('latest {}'.format(latest))
if latest is not None:
    model.load_weights(latest)
    print('load weights completed')

    predictions = model(x_test[:1]).numpy()
    print(predictions)
    print('{0:.10f}'.format(loss_fn(y_test[:1], predictions).numpy()))
    print('print prediction of load weight model completed')

# Train model, epochs is number of traing iterations
model.fit(
    x_train,
    y_train,
    epochs=3,
    callbacks=[cp_callback],  # Pass callback to training
)
print('train model completed')

predictions = model(x_test[:1]).numpy()
print(predictions)
print('{0:.10f}'.format(loss_fn(y_test[:1], predictions).numpy()))
print('print prediction of trained model completed')

# Check performance
# print(model.evaluate(x_train,  y_train, verbose=2))
print(model.evaluate(x_test, y_test, verbose=2))
print('check model performance completed')

# Print result?
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
print(probability_model(x_test[:5]))
print('print probabilities completed')

print(os.listdir(checkpoint_dir))
print('print checkpoints completed')

# Save the entire model as a `.keras` zip archive.
model.save('my_model.keras')
print('save model completed')
