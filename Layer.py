from Neuron import Neuron


class Layer:
    def __init__(self, number_of_neurons, input_size, activation_function):
        self.outputs = []
        self.neurons = [Neuron(input_size, activation_function) for i in range(number_of_neurons)]

    def forward(self, inputs):
        self.outputs = [neuron.calculate_output(inputs) for neuron in self.neurons]
        return self.outputs
