# from keras.datasets import mnist
import keras.src.datasets.mnist as mnist
import numpy as np

import lib

(images, labels), (testing_images, testing_labels) = mnist.load_data()

mnist_image = images[0]
for row in mnist_image:
    print("".join([" " if p == 0 else "@" for p in row]))

print(np.shape(images))
print(np.shape(labels))

# HYPER PARAMS
EPOCHS = 3
LEARNING_RATE = 0.1
BATCH_SIZE = 10

# we need weights and biases for each layer
# activation function, loss function,
# gradient descent can be performed by calculating the derivative of the loss function and minimizing the slope.


class NNLayer:
    weights = None
    biases = None
    nodes = None
    node_count = 0
    previous_layer_count = 0

    def __init__(self, previous_layer_count, nodes=None, node_count=10):
        if nodes is not None:
            self.nodes = nodes.flatten()
            self.node_count = self.nodes.size
            return
        self.node_count = node_count
        self.previous_layer_count = previous_layer_count
        self.weights = np.random.uniform(-1.0, 1.0, self.previous_layer_count)
        self.biases = np.random.uniform(0, 0.001, self.node_count)
        self.nodes = np.zeros(self.node_count)

    def feed_forward(self, input_nodes, activation_function):
        if input_nodes.size != self.previous_layer_count:
            raise Exception("Number of Weights and Biases should match previous input size")
        i = 0
        linear_combination = 0
        while i < self.node_count:
            linear_combination = np.vecdot(input_nodes, self.weights) + self.biases[i]
            self.nodes[i] = linear_combination
            i += 1
        self.nodes = activation_function(self.nodes)
        print("activation?", self.nodes)
        return self.nodes

    def back_prop(self):
        pass


class NeuralNet:
    input_set = []
    input_layer = None
    hidden_layers = []
    output = None
    forward_outputs = []

    def __init__(self, input_set, hidden_layer_count=2, hidden_layer_size=16, output_size=9):
        self.input_set = input_set
        prev_layer_size = self.input_set[0].flatten().size
        for i in range(hidden_layer_count):
            hl = NNLayer(prev_layer_size, node_count=hidden_layer_size)
            self.hidden_layers.append(hl)
            prev_layer_size = hl.node_count
        self.output = NNLayer(prev_layer_size, node_count=output_size)

    def forward(self):
        # we need read every image of our input set
        # TODO: potentially do a matrix calculation for input
        for inp in self.input_set[:5]:
            # create layer for current image we are reading
            self.input_layer = NNLayer(0, inp)

            # capture first hidden layer output
            curr_layer = 0
            previous_layer_output = self.hidden_layers[curr_layer].feed_forward(self.input_layer.nodes, lib.leaky_ReLU)

            # for the rest of the hidden layers, call feed_forward based on previous output
            while curr_layer < (len(self.hidden_layers) - 1):
                curr_layer += 1
                curr_layer_output = self.hidden_layers[curr_layer].feed_forward(previous_layer_output, lib.leaky_ReLU)
                previous_layer_output = curr_layer_output
            print(previous_layer_output)
            final_output = self.output.feed_forward(previous_layer_output, lib.softmax)
            self.forward_outputs.append(final_output)


# input layer (784 nodes, one for each pixel)
images_flattened = [image.flatten() for image in images]
image_size = images_flattened[0].size

nn = NeuralNet(images_flattened)

nn.forward()

print(nn.forward_outputs)


# input_l = NNLayer(0, nodes=images_flattened[0])
# hidden1 = NNLayer(image_size, node_count=16)
# hidden2 = NNLayer(hidden1.node_count, node_count=16)
# output = NNLayer(hidden2.node_count, node_count=9)

# print(hidden1.weights)
# print(hidden1.biases)
# print(hidden2.weights, hidden2.biases)
# print(output.weights, output.biases)


# h1_out = hidden1.feed_forward(input_l.nodes, lib.leaky_ReLU)
# h2_out = hidden2.feed_forward(
#     h1_out,
#     lib.leaky_ReLU
# )
# out1 = output.feed_forward(
#     h2_out,
#     lib.sigmoid
# )


# for img in images_flattened:

# hidden layers (2)

# output layer (one-hot encoded for each number 0-9)
