import random
import numpy as np
import Activations


class Neuron:
    def __init__(self, input_size, activation_function):
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = random.uniform(-1, 1)
        self.activation_function = activation_function
        self.output = 0
        self.delta = 0

    def calculate_output(self, inputs):
        weighted_sum = self._calculate_weighted_sum(inputs)
        self.output = self.activation_function(weighted_sum)
        return self.output

    def _calculate_weighted_sum(self, inputs):
        return sum(weight * inp for weight, inp in zip(self.weights, inputs)) + self.bias

    def calculate_derivative(self):
        if self.activation_function == Activations.sigmoid_activation:
            return self.output * (1 - self.output)
        elif self.activation_function == Activations.linear_activation:
            return 1
