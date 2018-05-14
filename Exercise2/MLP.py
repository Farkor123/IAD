#!/usr/bin/python3
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
        if activator == 'leakyrelu':
            self.activation = self.__leakyrelu
            self.derivative = self.__leakyrelu_derivative
        elif activator == 'relu':
            self.activation = self.__relu
            self.derivative = self.__relu_derivative
        elif activator == 'sigmoid':
            self.activation = self.__sigmoid
            self.derivative = self.__sigmoid_derivative
        elif activator == 'softsign':
            self.activation = self.__softsign
            self.derivative = self.__softsign_derivative
        elif activator == 'tanh':
            self.activation = self.__tanh
            self.derivative = self.__tanh_derivative

    @staticmethod
    def __leakyrelu(x):
        z, y = x.shape
        d = np.asmatrix(np.zeros((z, y)))
        for _ in range(z):
            for __ in range(y):
                if x[_, __] < 0.0:
                    d[_, __] = 0.01 * x[_, __]
                else:
                    d[_, __] = x[_, __]
        return d

    @staticmethod
    def __relu(x):
        z, y = x.shape
        d = np.asmatrix(np.zeros((z, y)))
        for _ in range(z):
            for __ in range(y):
                if x[_, __] < 0.0:
                    d[_, __] = 0.0
                else:
                    d[_, __] = x[_, __]
        return d

    @staticmethod
    def __sigmoid(x):
        z, y = x.shape
        d = np.asmatrix(np.zeros((z, y)))
        for _ in range(z):
            for __ in range(y):
                d[_, __] = (1.0 / (1.0 + np.exp(-x[_, __])))
        return d

    @staticmethod
    def __softsign(x):
        z, y = x.shape
        d = np.asmatrix(np.zeros((z, y)))
        for _ in range(z):
            for __ in range(y):
                d[_, __] = (x[_, __] / (1.0 + np.abs(x[_, __])))
        return d

    @staticmethod
    def __tanh(x):
        z, y = x.shape
        d = np.asmatrix(np.zeros((z, y)))
        for _ in range(z):
            for __ in range(y):
                d[_, __] = (np.exp(x[_, __]) - np.exp(-x[_, __])) / (np.exp(x[_, __]) + np.exp(-x[_, __]))
        return d

    @staticmethod
    def __leakyrelu_derivative(x):
        z, y = x.shape
        d = np.asmatrix(np.zeros((z, y)))
        for _ in range(z):
            for __ in range(y):
                if x[_, __] < 0:
                    d[_, __] = 0.01
                else:
                    d[_, __] = 1
        return d

    @staticmethod
    def __relu_derivative(x):
        z, y = x.shape
        d = np.asmatrix(np.zeros((z, y)))
        for _ in range(z):
            for __ in range(y):
                if x[_, __] < 0.0:
                    d[_, __] = 0.0
                else:
                    d[_, __] = 1.0
        return d

    @staticmethod
    def __sigmoid_derivative(x):
        z, y = x.shape
        d = np.asmatrix(np.zeros((z, y)))
        for _ in range(z):
            for __ in range(y):
                d[_, __] = x[_, __] * (1.0 - x[_, __])
        return d

    @staticmethod
    def __softsign_derivative(x):
        z, y = x.shape
        d = np.asmatrix(np.zeros((z, y)))
        for _ in range(z):
            for __ in range(y):
                d[_, __] = (x[_, __] / ((1.0 + np.abs(x[_, __])) * (1.0 + np.abs(x[_, __]))))
        return d

    @staticmethod
    def __tanh_derivative(x):
        z, y = x.shape
        d = np.asmatrix(np.zeros((z, y)))
        for _ in range(z):
            for __ in range(y):
                d[_, __] = 1.0 - (x[_, __] * x[_, __])
        return d


class NeuralNetwork:
    def __init__(self, layers_array):
        self.layers = layers_array

    def forward_propagate(self, input_matrix):
        self.layers[0].input = self.layers[0].weights * input_matrix + self.layers[0].bias
        self.layers[0].output = self.layers[0].activation(self.layers[0].input)
        for i in range(1, self.layers.size):
            self.layers[i].input = self.layers[i].weights * self.layers[i - 1].output + self.layers[i].bias
            self.layers[i].output = self.layers[i].activation(self.layers[i].input)

    def compute_errors(self, input_matrix, output_matrix):
        self.forward_propagate(input_matrix)
        self.layers[-1].error = output_matrix - self.layers[-1].output
        for i in reversed(range(self.layers.size - 1)):
            self.layers[i].error = self.layers[i + 1].weights.T * np.diagflat(
                self.layers[i+1].derivative(self.layers[i + 1].output)) * self.layers[i + 1].error

    def back_propagate(self, input_matrix, output_matrix, _lambda, _momentum):
        self.compute_errors(input_matrix, output_matrix)
        self.layers[0].v = _lambda * np.multiply(self.layers[0].derivative(self.layers[0].output),
                                                     self.layers[0].error) * input_matrix.T + _momentum * self.layers[
                               0].v
        self.layers[0].vb = _lambda * np.multiply(self.layers[0].derivative(self.layers[0].output),
                                                      self.layers[0].error) + _momentum * self.layers[0].vb
        self.layers[0].weights = self.layers[0].weights + self.layers[0].v
        self.layers[0].bias = self.layers[0].bias + self.layers[0].vb
        for i in range(1, self.layers.size):
            self.layers[i].v = _lambda * np.multiply(self.layers[i].derivative(self.layers[i].output),
                                                         self.layers[i].error) * self.layers[
                                   i - 1].output.T + _momentum * self.layers[i].v
            self.layers[i].vb = _lambda * np.multiply(self.layers[i].derivative(self.layers[i].output),
                                                          self.layers[i].error) + _momentum * self.layers[i].vb
            self.layers[i].weights = self.layers[i].weights + self.layers[i].v
            self.layers[i].bias = self.layers[i].bias + self.layers[i].vb
