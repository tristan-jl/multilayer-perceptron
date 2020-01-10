import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from Network.Layers import Dense, ReLU
from Network.utils import load_dataset, plot_sample, iterate_minibatches, predict, train

plt.style.use("dark_background")


def main():
    train_data, train_labels, test_data, test_labels = load_dataset(flatten=True)
    network = [
        Dense(train_data.shape[1], 100),
        ReLU(),
        Dense(100, 200),
        ReLU(),
        Dense(200, 10),
    ]

    plot_sample(train_data, train_labels)
    print(test_labels)

    train_log = []
    val_log = []
    for epoch in range(25):
        for x_batch, y_batch in iterate_minibatches(
            train_data, train_labels, batchsize=32, shuffle=True
        ):
            train(network, x_batch, y_batch)

        train_log.append(np.mean(predict(network, train_data) == test_data))
        val_log.append(np.mean(predict(network, test_data) == test_labels))

        clear_output()
        print("Epoch", epoch)
        print("Train accuracy:", train_log[-1])
        print("Val accuracy:", val_log[-1])
        plt.plot(train_log, label="train accuracy")
        plt.plot(val_log, label="val accuracy")
        plt.legend(loc="best")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main()
