import keras
import matplotlib.pyplot as plt
from Layers.mp_layers import Layer

def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.astype(float) / 255.0
    X_test = X_test.astype(float) / 255.0
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])
    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)
plt.figure(figsize=[6, 6])
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.title("Label: %i" % y_train[i])
    plt.imshow(X_train[i].reshape([28, 28]), cmap="gray")

network = [Dense(X_train.shape[1], 100), ReLU(), Dense(100, 200), ReLU(), Dense(200, 10)]


def forward(network, X):
    activations = []
    input = X
    for l in network:
        activations.append(l.forward(input))
        input = activations[-1]

    assert len(activations) == len(network)
    return activations


def predict(network, X):
    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)


def train(network, X, y):
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations  # layer_input[i] is an input for network[i]
    logits = layer_activations[-1]

    loss = softmax_crossentropy_with_logits(logits, y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits, y)

    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]

        loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)  # grad w.r.t. input, also weight updates

    return np.mean(loss)
