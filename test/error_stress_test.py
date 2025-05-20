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
epsilon = 1e-4

(X_train, y_train), _ = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1)
X_train = X_train.astype('float32') / 255

y_train = tf.keras.utils.to_categorical(y_train, 10)

for trial in range(25):
    print(f"# hidden layers: {trial}")
    layers = [Layer(28*28, 50, Lineal())]
    for i in range(trial):
        layers.append(Layer(50, 50, LeakyReLU()))

    layers.append(Layer(50, n_classes, Lineal()))

    loss_function = SoftmaxCrossEntropy()
    mlp = MLP(layers, loss_function, learning_rate=0.01)
    # mlp.visualize()

    ri = np.random.randint(0, len(X_train))
    x = X_train[ri]    
    y_true = y_train[ri]
    
    y_pred = mlp.forward(x)
    mlp.backprop(y_pred, y_true)

    weights_gradient = np.array([None for _ in layers])
    biases_gradient = np.array([None for _ in layers])
    for i, layer in enumerate(layers):
        weights_gradient[i] = layer.outweights_gradient
        biases_gradient[i] = layer.biases_gradient

    numerical_weights_gradient = np.array([None for _ in layers])
    numerical_biases_gradient = np.array([None for _ in layers])
    for l, layer in enumerate(layers):
        numerical_biases_gradient[l] = np.zeros_like(layer.biases)
        
        for i in range(len(layer.biases)):
            layer.biases[i] += epsilon
            plus_loss = loss_function.get_loss(mlp.forward(x), y_true)

            layer.biases[i] -= 2*epsilon
            minus_loss = loss_function.get_loss(mlp.forward(x), y_true)

            numerical_biases_gradient[l][i] = (plus_loss - minus_loss)/(2*epsilon)
            
            layer.biases[i] += epsilon

        if layer.outweights is None:
            continue

        numerical_weights_gradient[l] = np.zeros_like(layer.outweights)
        for i in range(len(layer.outweights)):
            for j in range(len(layer.outweights[0])):
                layer.outweights[i][j] += epsilon
                plus_loss = loss_function.get_loss(mlp.forward(x), y_true)
                
                layer.outweights[i][j] -= 2*epsilon
                minus_loss = loss_function.get_loss(mlp.forward(x), y_true)
                
                numerical_weights_gradient[l][i][j] = (plus_loss - minus_loss)/(2*epsilon)
                layer.outweights[i][j] += epsilon

    biases_numerator = 0
    biases_denominator = 0

    weights_numerator = 0
    weights_denominator = 0
    for i in range(len(layers)):
        biases_numerator += sum(abs(biases_gradient[i] - numerical_biases_gradient[i]))
        biases_denominator += sum(abs(biases_gradient[i]) + abs(numerical_biases_gradient[i]))

        if layers[i].outweights is None:
            continue
        weights_numerator += sum(sum(abs(weights_gradient[i] - numerical_weights_gradient[i])))
        weights_denominator += sum(sum(abs(weights_gradient[i]) + abs(numerical_weights_gradient[i])))

    print("Biases Diff:", biases_numerator/biases_denominator)
    print("Weights Diff:", weights_numerator/weights_denominator)