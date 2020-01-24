import pickle
import numpy as np
import matplotlib.pyplot as plt
from Network.layers import Dense, ReLU
from Network.utils import load_dataset

train_data, train_labels, test_data, test_labels = load_dataset(flatten=True)


def load_network():
    try:
        file_name = "../dist/network_file.p"
        with open(file_name, 'rb') as pickled:
            data = pickle.load(pickled)
            network = data['network']
        return network
    except FileNotFoundError:
        raise Exception("Could not find model file")


network = load_network()

data = test_data
labels = test_labels
predictions = np.zeros(labels.shape)
# for i in range(len(data)):
#     predictions[i] = predict(network, data[i])
#
# for i in range(len(predictions)):
#     print(labels[i], predictions[i])

plt.figure(figsize=[6, 6])
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.title(f"Label: {labels[i]}, Prediction: {network.predict(data[i])}")
    plt.imshow(data[i].reshape([28, 28]), cmap="gray")

plt.tight_layout()
plt.show()

np.save("a", data[0])
