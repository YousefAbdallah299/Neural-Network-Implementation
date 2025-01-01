from Layer import Layer
import numpy as np
import Activations


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, epochs):
        self.epochs = epochs
        self._network_layers = self._setup_layers(input_size, hidden_size, output_size)
        self.learning_rate = learning_rate

    def _setup_layers(self, input_dim, hidden_dim, output_dim):
        input_layer = Layer(input_dim, 0, None)
        hidden_layer = Layer(hidden_dim, input_dim, Activations.sigmoid_activation)
        output_layer = Layer(output_dim, hidden_dim, Activations.linear_activation)

        return [input_layer, hidden_layer, output_layer]

    def _forward_pass(self, input_data: np.ndarray):
        self._network_layers[0].outputs = input_data

        for layer in self._network_layers[1:]:
            input_data = layer.forward(input_data)

        return input_data

    def _update_output_layer(self, target_output):
        output_layer = self._network_layers[2]
        hidden_layer_outputs = self._network_layers[1].outputs

        for neuron in output_layer.neurons:
            neuron.delta = neuron.calculate_derivative() * (target_output - neuron.output)
            for i, weight in enumerate(neuron.weights):
                neuron.weights[i] += self.learning_rate * neuron.delta * hidden_layer_outputs[i]
            neuron.bias += self.learning_rate * neuron.delta

        return [neuron.delta for neuron in output_layer.neurons], [neuron.weights for neuron in
                                                                      output_layer.neurons]

    def _update_hidden_layer(self, output_deltas, output_weights):
        hidden_layer = self._network_layers[1]
        input_layer_outputs = self._network_layers[0].outputs

        for i, neuron in enumerate(hidden_layer.neurons):
            neuron.delta = neuron.calculate_derivative() * sum(
                output_deltas[j] * output_weights[j][i] for j in range(len(output_deltas)))
            for j, weight in enumerate(neuron.weights):
                neuron.weights[j] += self.learning_rate * neuron.delta * input_layer_outputs[j]
            neuron.bias += self.learning_rate * neuron.delta

    def _backward_propagate(self, target_output):
        output_deltas, output_weights = self._update_output_layer(target_output)
        self._update_hidden_layer(output_deltas, output_weights)

    def _single_prediction(self, input_data):
        return self._forward_pass(input_data)[0]

    def _calculate_error(self, predicted, actual, scaler):
        if scaler is not None:
            predicted = scaler.inverse_transform(np.array(predicted).reshape(1, -1))
            actual = scaler.inverse_transform(actual.reshape(1, -1))
        return np.sum((predicted - actual) ** 2)









#public functions to be called in main
    def train(self, input, output, acceptable_error, scaler_y):
        length = len(input)
        for epoch in range(self.epochs):
            error = 0.0
            for i in range(length):
                temp = self._forward_pass(input[i])
                self._backward_propagate(output[i])
                error += self._calculate_error(temp, output[i], scaler_y)

            error *= 0.5
            error /= length
            print("Epoch: ", epoch + 1, " Error in this epoch: ", error)
            if error < acceptable_error:
                break

    def predict(self, test_inputs):
        return (np.array(
            [self._single_prediction(input_data) for input_data in test_inputs])
                .reshape(-1, 1))

