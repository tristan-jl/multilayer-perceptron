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
