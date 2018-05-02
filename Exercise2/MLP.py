#!/usr/bin/python3
import numpy as np
from mpmath import mp
import matplotlib.pyplot as plt
from tqdm import tqdm

class neuronlayer():
	def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
		''' weights matrix, outputs x inputs '''
		self.weights = (2 * np.asmatrix(np.random.random((number_of_neurons, number_of_inputs_per_neuron))) - 1) / 2
		#bias matrix, outputs x 1
		self.bias = (2 * np.asmatrix(np.random.random((number_of_neurons, 1))) - 1) / 2
		#self.bias = np.zeros((number_of_neurons, 1))
		#bias change matrix
		self.vb = np.asmatrix(np.zeros((number_of_neurons, 1)))
		#output matrix, activation(input)
		self.output = np.matrix([])
		#input matrix, wieghts x last_layer_output + bias
		self.input = np.matrix([])
		#error matrix, outputs x 1
		self.error = np.matrix([])
		#matrix of weights deltas + momentum * previous_deltas
		self.v = np.asmatrix(np.zeros((number_of_neurons, number_of_inputs_per_neuron)))

class neuralnetwork():
	def __init__(self, layers_array):
		self.layers = layers_array

	def __sigmoid(self, x):
		z, y = x.shape
		d = np.asmatrix(np.zeros((z, y)))
		for _ in range(z):
			for __ in range(y):
				d[_, __] = (1.0 / (1.0 + np.exp(-x[_, __])))
		return d

	def __tanh(self, x):
		z, y = x.shape
		d = np.zeros((z, y))
		for _ in range(z):
			for __ in range(y):
				d[_, __] = (mp.exp(x[_, __]) - mp.exp(-x[_, __])) / (mp.exp(x[_, __]) + mp.exp(-x[_, __]))
		return d

	def __sigmoid_derivative(self, x):
		z, y = x.shape
		d = np.asmatrix(np.zeros((z, y)))
		for _ in range(z):
			for __ in range(y):
				d[_, __] = x[_, __] * (1.0 - x[_, __])
		return d

	def __tanh_derivative(self, x):
		z, y = x.shape
		d = np.zeros((z, y))
		for _ in range(z):
			for __ in range(y):
				d[_, __] = 1 - x[_, __] * x[_, __]
		return d

	def forward_propagate(self, input_matrix):
		self.layers[0].input = self.layers[0].weights * input_matrix + self.layers[0].bias
		self.layers[0].output = self.__sigmoid(self.layers[0].input)
		for i in range(1, self.layers.size):
			self.layers[i].input = self.layers[i].weights * self.layers[i-1].output + self.layers[i].bias
			self.layers[i].output = self.__sigmoid(self.layers[i].input)

	def compute_errors(self, input_matrix, output_matrix):
		self.forward_propagate(input_matrix)
		self.layers[-1].error = output_matrix - self.layers[-1].output
		for i in reversed(range(self.layers.size - 1)):
			self.layers[i].error = self.layers[i+1].weights.T * np.diagflat(self.__sigmoid_derivative(self.layers[i+1].output)) * self.layers[i+1].error

	def back_propagate(self, input_matrix, output_matrix, _lambda, _momentum):
		self.compute_errors(input_matrix, output_matrix)
		self.layers[0].v = 2 * _lambda * np.multiply(self.__sigmoid_derivative(self.layers[0].output), self.layers[0].error) * input_matrix.T + _momentum * self.layers[0].v
		self.layers[0].vb = 2 * _lambda * np.multiply(self.__sigmoid_derivative(self.layers[0].output), self.layers[0].error) + _momentum * self.layers[0].vb
		self.layers[0].weights = self.layers[0].weights + self.layers[0].v
		self.layers[0].bias = self.layers[0].bias + self.layers[0].vb
		for i in range(1, self.layers.size):
			self.layers[i].v = 2 * _lambda * np.multiply(self.__sigmoid_derivative(self.layers[i].output), self.layers[i].error) * self.layers[i-1].output.T + _momentum * self.layers[i].v
			self.layers[i].vb = 2 * _lambda * np.multiply(self.__sigmoid_derivative(self.layers[i].output), self.layers[i].error) + _momentum * self.layers[i].vb
			self.layers[i].weights = self.layers[i].weights + self.layers[i].v
			self.layers[i].bias = self.layers[i].bias + self.layers[i].vb
