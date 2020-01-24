import pickle

import matplotlib.pyplot as plt
import numpy as np

from Network.layers import Dense, ReLU
from Network.network import Network
from Network.utils import load_dataset, iterate_minibatches

plt.style.use("dark_background")


def main():
    train_data, train_labels, test_data, test_labels = load_dataset(flatten=True)
    network = Network([
        Dense(train_data.shape[1], 100),
        ReLU(),
        Dense(100, 200),
        ReLU(),
        Dense(200, 10),
    ])

    train_log = []
    val_log = []
    plt.show()
    for epoch in range(25):
        for x_batch, y_batch in iterate_minibatches(
            train_data, train_labels, batchsize=32, shuffle=True
        ):
            network.train(x_batch, y_batch)

        train_log.append(np.mean(network.predict(train_data) == train_labels))
        val_log.append(np.mean(network.predict(test_data) == test_labels))

        print("Epoch", epoch + 1)
        print("Train accuracy:", train_log[-1])
        print("Val accuracy:", val_log[-1])

        plt.plot(train_log, label="train accuracy")
        plt.plot(val_log, label="val accuracy")
        plt.draw()
        plt.pause(0.001)

    pickle.dump({"network": network}, open("../dist/network_file" + ".p", "wb"))
    plt.show()


if __name__ == "__main__":
    main()
