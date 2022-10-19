import numpy as np

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class LayerDense:
    def __init__(self, n_inputs, n_neurons):

        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = LayerDense(4, 6)
layer2 = LayerDense(6, 2)

layer1.forward(X)
print(layer1.output)
print("\n")
layer2.forward(layer1.output)
print(layer2.output)
# TODO: document this code