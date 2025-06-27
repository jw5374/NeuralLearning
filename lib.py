import numpy as np


# ********** ACTIVATION FUNCTIONS **********
def reLU(input):
    # derivative: 0 if x < 0, 1 if x > 0
    return np.maximum(0, input)


def leaky_ReLU(input):
    # derivative: 0.01 if x < 0, 1 if x > 0
    return np.maximum((0.01 * input), input)


def sigmoid(input):
    # derivative: sigmoid(input) * (1 - sigmoid(input))
    return 1 / (1 + np.exp(input * -1))


def softmax(input):
    return np.exp(input) / float(np.sum(np.exp(input)))


def tanh(input):
    # derivative: 1 - tanh(input)^2
    return (np.exp(input) - np.exp(input * - 1)) / (np.exp(input) + np.exp(input * - 1))


# ********** LOSS FUNCTIONS **********
def mse(input, expected):
    return
