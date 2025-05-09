import numpy as np


class ActFunction:
    def apply(self, x: np.array):
        raise NotImplementedError("Apply activation function has not been implemented!")

    def derivative(self, x: np.array):
        raise NotImplementedError("Derivative of the activation function has not been implemented!")


class Sigmoid(ActFunction):
    def apply(self, x: np.array):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.array):
        sig = self.apply(x)
        return  sig * (1 - sig)
    

class Lineal(ActFunction):
    def apply(self, x: np.array):
        return x

    def derivative(self, x: np.array):
        return 1
    

class ReLU(ActFunction):
    def apply(self, x: np.array):
        return np.maximum(0, x)

    def derivative(self, x: np.array):
        return np.where(x > 0, 1, 0)


class LeakyReLU(ActFunction):
    def apply(self, x: np.array):
        return np.maximum(0.01 * x, x)
    
    def derivative(self, x: np.array):
        return np.where(x > 0, 1, 0.01)


class Tanh(ActFunction):
    def apply(self, x: np.array):
        return np.tanh(x)

    def derivative(self, x: np.array):
        return 1 - self.apply(x) ** 2


