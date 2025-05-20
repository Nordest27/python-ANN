import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from ann import MLP, Layer
from activation_functions import Lineal, LeakyReLU
from loss_functions import SoftmaxCrossEntropy
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

n_classes = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

layers = [
    Layer(28*28, 100, Lineal()),
    Layer(100, 50, LeakyReLU()),
    Layer(50, 10, LeakyReLU()),
    Layer(10, 0, Lineal())
]
mlp = MLP(layers, SoftmaxCrossEntropy(), learning_rate=0.01)

for it in range(15):
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    acc_loss = 0
    correct_predictions = 0
    for i in range(len(X_train)):
        x = X_train[i]    
        y_true = y_train[i]
        y_pred = mlp.forward(x)
        if np.argmax(y_true) == np.argmax(y_pred):
            correct_predictions += 1
        acc_loss += mlp.backprop(y_pred,  y_true)
        if i%10 == 0:
            mlp.update()
    print(f"It: {it} Loss: {acc_loss/len(X_train):.10f}, Acc: {correct_predictions/len(X_train):.2f}")


total_error = 0
correct_predictions = 0
for i in range(len(X_test)):
    x = X_test[i]
    y_true = y_test[i]
    y_pred = mlp.forward(x)
    if np.argmax(y_true) == np.argmax(y_pred):
        correct_predictions += 1
    error = mlp.loss_f.get_loss(y_pred, y_true)
    total_error += error

average_error = total_error / len(X_test)
print("Average error on test data:", average_error)
print("Accuracy:", correct_predictions/len(X_test))

