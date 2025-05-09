import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ann import MLP, Layer
from activation_functions import Sigmoid, Lineal, ReLU, Tanh, LeakyReLU
from loss_functions import MeanSquaredError, SigmoidCrossEntropy, SoftmaxCrossEntropy
import numpy as np

n_classes = 2

X, y = make_classification(n_samples=1000, n_features=20, n_classes=n_classes, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

layers = [
    Layer(20, 10, Lineal()),
    Layer(10, 1, LeakyReLU()),
    Layer(1, 0, Lineal())
]
mlp = MLP(layers, SigmoidCrossEntropy(), learning_rate=0.01)

for _ in range(1000):
    acc_loss = 0
    correct_predictions = 0
    for i in range(len(X_train)):
        x = X_train[i]
        y_true = np.array([y_train[i]])
        y_pred = mlp.forward(x)
        if (y_pred[0] <= 0.5 and y_true[0] < 0.5) or (y_pred[0] >= 0.5 and y_true[0] > 0.5):
            correct_predictions += 1
        acc_loss += mlp.backprop(y_pred, y_true)
        if i%10 == 0:
            mlp.update()
    print(f"Loss: {acc_loss/len(X_train):.10f}, Acc: {correct_predictions/len(X_train):.2f}")


correct_predictions = 0
total_error = 0
for i in range(len(X_test)):
    x = X_test[i]
    y_true = np.array([y_test[i]])
    y_pred = mlp.forward(x)
    if (y_pred[0] <= 0.5 and y_true[0] < 0.5) or (y_pred[0] >= 0.5 and y_true[0] > 0.5):
        correct_predictions += 1
    error = mlp.loss_f.get_loss(y_pred, y_true)
    total_error += error

average_error = total_error / len(X_test)
print("Average error on test data:", average_error)
print("Accuracy:", correct_predictions/len(X_test))


