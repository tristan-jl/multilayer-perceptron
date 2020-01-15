import numpy as np
import matplotlib.pyplot as plt
from Network.Layers import Dense, ReLU
from Network.utils import load_dataset, plot_sample, iterate_minibatches, predict, train

plt.style.use("dark_background")


def main():
    train_data, train_labels, test_data, test_labels = load_dataset(flatten=True)
    print(train_data.shape[1])
    network = [
        Dense(train_data.shape[1], 100),
        ReLU(),
        Dense(100, 200),
        ReLU(),
        Dense(200, 10),
    ]

    # plot_sample(train_data, train_labels)
    # print(test_labels)

    train_log = []
    val_log = []
    plt.show()
    for epoch in range(25):
        for x_batch, y_batch in iterate_minibatches(
            train_data, train_labels, batchsize=32, shuffle=True
        ):
            train(network, x_batch, y_batch)

        train_log.append(np.mean(predict(network, train_data) == train_labels))
        val_log.append(np.mean(predict(network, test_data) == test_labels))

        print("Epoch", epoch + 1)
        print("Train accuracy:", train_log[-1])
        print("Val accuracy:", val_log[-1])

        plt.plot(train_log, label="train accuracy")
        plt.plot(val_log, label="val accuracy")
        # plt.legend(loc="best")
        plt.draw()
        plt.pause(0.001)

    np.save("../weights/0.npy", network[0].weights)
    np.save("../weights/2.npy", network[2].weights)
    np.save("../weights/4.npy", network[4].weights)
    plt.show()


if __name__ == "__main__":
    main()
