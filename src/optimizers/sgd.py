import numpy as np


class SGD:
    def __init__(self, parameters, learning_rate):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.parameters = list(parameters)
        self.learning_rate = learning_rate

    def step(self):
        for parameter in self.parameters:
            parameter.data -= self.learning_rate * parameter.grad

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = np.zeros_like(parameter.data)
