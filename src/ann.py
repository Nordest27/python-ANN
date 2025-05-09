import numpy as np
from activation_functions import ActFunction
from loss_functions import LossFunction


class Layer:

    def __init__(
            self, 
            n_nodes: int, 
            n_next_nodes: int,
            act_f: ActFunction
        ):
        self.shape = (n_next_nodes, n_nodes)
        
        self.values = np.zeros(n_nodes)
        self.local_derivative = np.zeros(n_nodes)

        self.biases = np.zeros(n_nodes)
        self.biases_update = np.zeros(n_nodes)

        self.outweights = np.random.random(self.shape)/100 if n_next_nodes else None
        self.outweights_update = np.zeros(self.shape) if n_next_nodes else None

        self.act_f = act_f
        self.update_count = 0

    def forward(self, values: np.array) -> np.array:
        self.values = values + self.biases
        self.local_derivative = self.act_f.derivative(self.values)
        self.values = self.act_f.apply(self.values)

        if self.outweights is not None:
            return np.dot(self.outweights, self.values)
        else:
            return self.values
    
    def backward(self, gradient: np.array) -> np.array:
        self.update_count += 1
        if self.outweights is not None: 
            self.outweights_update += np.outer(gradient, self.values)
            gradient = np.dot(np.transpose(self.outweights), gradient)

        self.biases_update += gradient
        gradient *= self.local_derivative

        return gradient

    def update(self, learning_rate: float):
        if self.update_count ==  0:
            print("Trying to update without values!")
            return
        if self.outweights is not None:
            self.outweights -= learning_rate * self.outweights_update / self.update_count
            self.outweights_update = np.zeros_like(self.outweights)

        self.biases -= learning_rate * self.biases_update / self.update_count
        self.biases_update = np.zeros_like(self.biases)

        self.update_count = 0
    
    def reset(self):
        self.values = np.zeros_like(self.values)
        self.local_derivative = np.zeros_like(self.values)

class MLP:

    layers: list[Layer]
    loss_f: LossFunction
    learning_rate: float

    def __init__(
            self, 
            layers: list[Layer], 
            loss_f: LossFunction,
            learning_rate: float,
        ):
        self.layers = layers
        self.loss_f = loss_f
        self.learning_rate = learning_rate

    def forward(self, x: np.array) -> np.array:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backprop(self, y_pred: np.array, y_true: np.array) -> np.array:
        loss, gradient = self.loss_f.get_loss_and_gradient(y_pred, y_true)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
            layer.reset()
        return loss
    
    def update(self):
        for layer in self.layers:
            layer.update(self.learning_rate)
