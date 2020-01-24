import numpy as np


class Network:
    def __init__(self, layers):
        self.network = layers

    def forward(self, input):
        activations = []
        for layer in self.network:
            activations.append(layer.forward(input))
            input = activations[-1]

        assert len(activations) == len(self.network)
        return activations

    def predict(self, x):
        logits = self.forward(x)[-1]
        return logits.argmax(axis=-1)

    def softmax_cross_entropy_with_logits(self, logits, reference_answers):
        logits_for_answers = logits[np.arange(len(logits)), reference_answers]
        cross_entropy = -logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))

        return cross_entropy

    def grad_softmax_cross_entropy_with_logits(self, logits, reference_answers):
        ones_for_answers = np.zeros_like(logits)
        ones_for_answers[np.arange(len(logits)), reference_answers] = 1

        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

        return (-ones_for_answers + softmax) / logits.shape[0]

    def train(self, x, y):
        layer_activations = self.forward(x)
        layer_inputs = [x] + layer_activations
        logits = layer_activations[-1]

        loss = self.softmax_cross_entropy_with_logits(logits, y)
        loss_grad = self.grad_softmax_cross_entropy_with_logits(logits, y)

        for layer_index in range(len(self.network))[::-1]:
            layer = self.network[layer_index]
            loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)

        return np.mean(loss)
