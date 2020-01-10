import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_dataset(flatten=False):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "../data/", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        "../data/", train=False, transform=transform, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    train_data, train_labels = next(iter(train_loader))
    test_data, test_labels = next(iter(test_loader))

    train_data = train_data.numpy()
    train_labels = train_labels.numpy()
    test_data = test_data.numpy()
    test_labels = test_labels.numpy()

    if flatten:
        train_data = train_data.reshape([train_data.shape[0], -1])
        test_data = test_data.reshape([test_data.shape[0], -1])
    return train_data, train_labels, test_data, test_labels


def plot_sample(data, labels):
    plt.figure(figsize=[6, 6])
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.title("Label: %i" % labels[i])
        plt.imshow(data[i].reshape([28, 28]), cmap="gray")

    plt.tight_layout()
    plt.show()


def softmax_cross_entropy_with_logits(logits, reference_answers):
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]
    cross_entropy = -logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))

    return cross_entropy


def grad_softmax_cross_entropy_with_logits(logits, reference_answers):
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (-ones_for_answers + softmax) / logits.shape[0]


def forward(network, x):
    activations = []
    input = x
    for layer in network:
        activations.append(layer.forward(input))
        input = activations[-1]

    assert len(activations) == len(network)
    return activations


def predict(network, x):
    logits = forward(network, x)[-1]
    return logits.argmax(axis=-1)


def train(network, x, y):
    layer_activations = forward(network, x)
    layer_inputs = [x] + layer_activations
    logits = layer_activations[-1]

    loss = softmax_cross_entropy_with_logits(logits, y)
    loss_grad = grad_softmax_cross_entropy_with_logits(logits, y)

    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]
        loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)

    return np.mean(loss)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx : start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
