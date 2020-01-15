import numpy as np
import matplotlib.pyplot as plt
from Network.Layers import Dense, ReLU
from Network.utils import load_dataset, plot_sample, iterate_minibatches, predict, train

train_data, train_labels, test_data, test_labels = load_dataset(flatten=True)
network = [
        Dense(train_data.shape[1], 100),
        ReLU(),
        Dense(100, 200),
        ReLU(),
        Dense(200, 10),
    ]

network[0].weights = np.load("../weights/0.npy")
network[2].weights = np.load("../weights/2.npy")
network[4].weights = np.load("../weights/4.npy")


data = test_data
labels = test_labels
predictions = np.zeros(labels.shape)
for i in range(len(data)):
    predictions[i] = predict(network, data[i])

for i in range(len(predictions)):
    print(labels[i], predictions[i])

# plt.figure(figsize=[6, 6])
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.title(f"Label: {labels[i]}, Prediction: {predict(network, data[i])}")
#     plt.imshow(data[i].reshape([28, 28]), cmap="gray")
#
# plt.tight_layout()
# plt.show()

