import numpy as np


class LossFunction:
    def get_loss(self, y_pred: np.array, y_true: np.array) -> np.array:
        raise NotImplementedError("Loss function has not been implemented!")

    def get_gradient(self, y_pred: np.array, y_true: np.array) -> np.array:
        raise NotImplementedError("Loss function has not been implemented!")
    
    def get_loss_and_gradient(
            self, 
            y_pred: np.array, 
            y_true: np.array
        ) -> tuple[np.array, np.array]:
        return (
            self.get_loss(y_pred, y_true), 
            self.get_gradient(y_pred, y_true)
        )


class MeanSquaredError(LossFunction):
    def get_loss(self, y_pred: np.array, y_true: np.array) -> np.array:
        return np.mean((y_pred - y_true) ** 2)

    def get_gradient(self, y_pred: np.array, y_true: np.array) -> np.array:
        return 2 * (y_pred - y_true) / len(y_pred)


class SigmoidCrossEntropy(LossFunction):
    def get_loss(self, logits, y_true):
        loss = np.sum(np.maximum(logits, 0) - logits * y_true + np.log(1 + np.exp(-np.abs(logits))))
        return loss

    def get_gradient(self, logits, y_true):
        # Sigmoid + simplified gradient
        sigmoid = 1 / (1 + np.exp(-logits))
        return sigmoid - y_true


class SoftmaxCrossEntropy(LossFunction):
    def get_loss(self, logits, y_true):
        logits_shifted = logits - np.max(logits)
        exps = np.exp(logits_shifted)
        probs = exps / np.sum(exps, axis=0)
        probs = np.clip(probs, 1e-12, 1. - 1e-12)
        return -np.sum(y_true * np.log(probs))

    def get_gradient(self, logits, y_true):
        logits_shifted = logits - np.max(logits)
        exps = np.exp(logits_shifted)
        probs = exps / np.sum(exps, axis=0)
        return probs - y_true

