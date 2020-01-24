import numpy as np
import pytest

from multilayer_perceptron.Network.layers import Layer, ReLU, Dense


@pytest.mark.parametrize(
    "input,expected",
    [(0, 0), ("abc", "abc"), (np.array([1, 2, 3]).all(), np.array([1, 2, 3]).all())],
)
def test_layer_forward(input, expected):
    layer = Layer()
    assert layer.forward(input) == expected


@pytest.mark.parametrize(
    "input,grad_output,expected",
    [
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[7, 8, 9], [10, 11, 12]]),
            np.array([[7, 8, 9], [10, 11, 12]]),
        ),
    ],
)
def test_layer_backward(input, grad_output, expected):
    layer = Layer()
    assert (layer.backward(input, grad_output) == expected).all()


@pytest.mark.parametrize("input,expected", [(0, 0), (10, 10), (-10, 0)])
def test_relu_forward(input, expected):
    relu = ReLU()
    assert relu.forward(input) == expected


@pytest.mark.parametrize(
    "input,grad_output,expected",
    [
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[7, 8, 9], [10, 11, 12]]),
            np.array([[7, 8, 9], [10, 11, 12]]),
        ),
    ],
)
def test_relu_backward(input, grad_output, expected):
    layer = Layer()
    assert (layer.backward(input, grad_output) == expected).all()


@pytest.mark.parametrize(
    "input_units,output_units,learning_rate", [(1, 1, 1), (10, 100, 5)]
)
def test_dense_constructor(input_units, output_units, learning_rate):
    dense = Dense(input_units, output_units, learning_rate)
    assert dense.learning_rate == learning_rate
    assert dense.weights.size == input_units * output_units
    assert (dense.bias == np.zeros(output_units)).all()


@pytest.mark.parametrize(
    "input,weights,bias,expected",
    [
        (
            np.array([0, 1, 2, 3]),
            np.array([4, 5, 6, 7]),
            np.array([8, 9, 10, 11]),
            np.array([46, 47, 48, 49]),
        ),
    ],
)
def test_dense_forward(input, weights, bias, expected):
    dense = Dense(2, 2)
    dense.weights = weights
    dense.bias = bias
    assert (dense.forward(input) == expected).all()
