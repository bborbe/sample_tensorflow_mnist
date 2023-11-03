#
# https://www.tensorflow.org/tutorials/quickstart/beginner
#
import os

# TensorFlow and tf.keras
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print(tf.__version__)

# Load a dataset
mnist = tf.keras.datasets.mnist

# x_train: uint8 NumPy array of grayscale image data with shapes
# y_train: uint8 NumPy array of digit labels (integers in range 0-9)
# x_test: uint8 NumPy array of grayscale image data with shapes
# y_test: uint8 NumPy array of digit labels (integers in range 0-9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print('load dataset completed')
model = tf.keras.models.load_model(
    'my_model.keras',
    compile=False,
)

# Show the model architecture
print(model.summary())
print('load model complete')

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Configure compiler
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
print('configure compiler completed')


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
