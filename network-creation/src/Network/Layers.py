import numpy as np


class Layer:
    def __init__(self):
        pass

    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        d_layer_d_input = np.eye(input.shape[1])
        return np.dot(grad_output, d_layer_d_input)


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return np.maximum(0, input)

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        super().__init__()
        self.learning_rate = learning_rate
        self.weights = np.random.normal(
            loc=0.0,
            scale=np.sqrt(2 / (input_units + output_units)),
            size=(input_units, output_units),
        )
        self.bias = np.zeros(output_units)

    def forward(self, input):
        return np.dot(input, self.weights) + self.bias

    def backward(self, input, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * input.shape[0]

        assert (
            grad_weights.shape == self.weights.shape
            and grad_biases.shape == self.bias.shape
        )

        self.weights = self.weights - self.learning_rate * grad_weights
        self.bias = self.bias - self.learning_rate * grad_biases

        return grad_input
