from numpy.typing import NDArray
import numpy as np


# ********** ACTIVATION FUNCTIONS **********
def reLU(input: NDArray[float | int]) -> NDArray[float]:
    # derivative: 0 if x < 0, 1 if x > 0
    return np.maximum(0, input)


def leaky_ReLU(input: NDArray[float | int]) -> NDArray[float]:
    # derivative: 0.01 if x < 0, 1 if x > 0
    return np.maximum((0.01 * input), input)


def sigmoid(input: NDArray[float | int]) -> NDArray[float]:
    # derivative: sigmoid(input) * (1 - sigmoid(input))
    return 1 / (1 + np.exp(input * -1))


def softmax(input: NDArray[float | int]) -> NDArray[float]:
    # prevent overflow of input in exponentiation
    subtract_max = input - np.max(input)
    return np.exp(subtract_max) / np.sum(np.exp(subtract_max))


def tanh(input: NDArray[float | int]) -> NDArray[float]:
    # derivative: 1 - tanh(input)^2
    return (np.exp(input) - np.exp(input * - 1)) / (np.exp(input) + np.exp(input * - 1))


# ********** LOSS FUNCTIONS **********
def mse(input: NDArray[float | int], expected: NDArray[float | int]) -> float:
    sum = 0
    for pred, targ in zip(input, expected):
        sum += np.square(targ - pred)
    return sum / input.size


def cross_entropy(input: NDArray[float], expected: NDArray[float]) -> float:
    # derivative with respect to any output neuron: (output of neuron at index i) - (target of neuron at index i)
    sum = 0
    for pred, targ in zip(input, expected):
        sum += np.multiply(targ, np.log(pred))
    return sum / -input.size
