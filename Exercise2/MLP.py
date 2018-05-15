import numpy as np


class NeuronLayer:
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron, activator):
        ''' weights matrix, outputs x inputs '''
        self.weights = (2 * np.asmatrix(np.random.random((number_of_neurons, number_of_inputs_per_neuron))).astype(np.float32) - 1) / 2
        # bias matrix, outputs x 1
        self.bias = (2 * np.asmatrix(np.random.random((number_of_neurons, 1))).astype(np.float32) - 1) / 2
        # self.bias = np.zeros((number_of_neurons, 1))
        # bias change matrix
        self.vb = np.asmatrix(np.zeros((number_of_neurons, 1)).astype(np.float32))
        # output matrix, activation(input)
        self.output = np.matrix([]).astype(np.float32)
        # input matrix, wieghts x last_layer_output + bias
        self.input = np.matrix([]).astype(np.float32)
        # error matrix, outputs x 1
        self.error = np.matrix([]).astype(np.float32)
        # matrix of weights deltas + momentum * previous_deltas
        self.v = np.asmatrix(np.zeros((number_of_neurons, number_of_inputs_per_neuron))).astype(np.float32)
        self.activator = activator

class NeuralNetwork:
    def __init__(self, layers_array):
        self.layers = layers_array

    def __activation_function(self, x, activate):
        z, y = x.shape
        d = np.asmatrix(np.zeros((z, y)))
        if activate == 'leakyrelu':
            for _ in range(z):
                for __ in range(y):
                    if x[_, __] < 0.0:
                        d[_, __] = 0.01 * x[_, __]
                    else:
                        d[_, __] = x[_, __]
            return d
        elif activate == 'relu':
            for _ in range(z):
                for __ in range(y):
                    if x[_, __] < 0.0:
                        d[_, __] = 0.0
                    else:
                        d[_, __] = x[_, __]
            return d
        elif activate == 'sigmoid':
            for _ in range(z):
                for __ in range(y):
                    d[_, __] = (1.0 / (1.0 + np.exp(-x[_, __])))
            return d
        elif activate == 'softsign':
            for _ in range(z):
                for __ in range(y):
                    d[_, __] = (x[_, __] / (1.0 + np.abs(x[_, __])))
            return d
        elif activate == 'tanh':
            for _ in range(z):
                for __ in range(y):
                    d[_, __] = (np.exp(x[_, __]) - np.exp(-x[_, __])) / (np.exp(x[_, __]) + np.exp(-x[_, __]))
            return d

    def __activation_function_derivative(self, x, activate):
        z, y = x.shape
        d = np.asmatrix(np.zeros((z, y)))
        if activate == 'leakyrelu':
            for _ in range(z):
                for __ in range(y):
                    if x[_, __] < 0:
                        d[_, __] = 0.01
                    else:
                        d[_, __] = 1
            return d
        elif activate == 'relu':
            for _ in range(z):
                for __ in range(y):
                    if x[_, __] < 0.0:
                        d[_, __] = 0.0
                    else:
                        d[_, __] = 1.0
            return d
        elif activate == 'sigmoid':
            for _ in range(z):
                for __ in range(y):
                    d[_, __] = x[_, __] * (1.0 - x[_, __])
            return d
        elif activate == 'softsign':
            for _ in range(z):
                for __ in range(y):
                    d[_, __] = (x[_, __] / ((1.0 + np.abs(x[_, __])) * (1.0 + np.abs(x[_, __]))))
            return d
        elif activate == 'tanh':
            for _ in range(z):
                for __ in range(y):
                    d[_, __] = 1.0 - (x[_, __] * x[_, __])
            return d

    def forward_propagate(self, input_matrix):
        self.layers[0].input = self.layers[0].weights * input_matrix + self.layers[0].bias
        self.layers[0].output = self.__activation_function(self.layers[0].input, self.layers[0].activator)
        for i in range(1, self.layers.size):
            self.layers[i].input = self.layers[i].weights * self.layers[i - 1].output + self.layers[i].bias
            self.layers[i].output = self.__activation_function(self.layers[i].input, self.layers[i].activator)

    def compute_errors(self, input_matrix, output_matrix):
        self.forward_propagate(input_matrix)
        self.layers[-1].error = output_matrix - self.layers[-1].output
        for i in reversed(range(self.layers.size - 1)):
            self.layers[i].error = self.layers[i + 1].weights.T * np.diagflat(
                self.__activation_function_derivative(self.layers[i + 1].output, self.layers[i + 1].activator)) * self.layers[i + 1].error

    def back_propagate(self, input_matrix, output_matrix, _lambda, _momentum):
        self.compute_errors(input_matrix, output_matrix)
        self.layers[0].v = _lambda * np.multiply(self.__activation_function_derivative(self.layers[0].output, self.layers[0].activator),
                                                     self.layers[0].error) * input_matrix.T + _momentum * self.layers[
                               0].v
        self.layers[0].vb = _lambda * np.multiply(self.__activation_function_derivative(self.layers[0].output, self.layers[0].activator),
                                                      self.layers[0].error) + _momentum * self.layers[0].vb
        self.layers[0].weights = self.layers[0].weights + self.layers[0].v
        self.layers[0].bias = self.layers[0].bias + self.layers[0].vb
        for i in range(1, self.layers.size):
            self.layers[i].v = _lambda * np.multiply(self.__activation_function_derivative(self.layers[i].output, self.layers[i].activator),
                                                         self.layers[i].error) * self.layers[
                                   i - 1].output.T + _momentum * self.layers[i].v
            self.layers[i].vb = _lambda * np.multiply(self.__activation_function_derivative(self.layers[i].output, self.layers[i].activator),
                                                          self.layers[i].error) + _momentum * self.layers[i].vb
            self.layers[i].weights = self.layers[i].weights + self.layers[i].v
            self.layers[i].bias = self.layers[i].bias + self.layers[i].vb
