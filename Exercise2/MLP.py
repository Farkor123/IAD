#!/usr/bin/python3
import numpy as np
import math as m
import matplotlib.pyplot as plt

class neuronlayer():
	def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
		#weights matrix, outputs x inputs
		self.weights = (2 * np.asmatrix(np.random.random((number_of_neurons, number_of_inputs_per_neuron))) - 1)
		#bias matrix, outputs x 1
		self.bias = (2 * np.asmatrix(np.random.random((number_of_neurons, 1))) - 1)
		#self.bias = np.zeros((number_of_neurons, 1))
		#bias change matrix
		self.vb = np.zeros((number_of_neurons, 1))
		#output matrix, activation(input)
		self.output = np.matrix([])
		#input matrix, wieghts x last_layer_output + bias
		self.input = np.matrix([])
		#error matrix, outputs x 1
		self.error = np.matrix([])
		#matrix of weights deltas + momentum * previous_deltas
		self.v = np.zeros((number_of_neurons, number_of_inputs_per_neuron))

class neuralnetwork():
	def __init__(self, layers_array):
		self.layers = np.array(layers_array)

	def __sigmoid(self, x):
		z, y = x.shape
		for _ in range(z):
			for __ in range(y):
				x[_, __] = (1 / (1 + np.exp(-x[_, __])))
		return x

	def __tanh(self, x):
		z, y = x.shape
		for _ in range(z):
			for __ in range(y):
				x[_, __] = (m.exp(x[_, __]) - m.exp(-x[_, __])) / (m.exp(x[_, __]) + m.exp(-x[_, __]))
		return x

	def __sigmoid_derivative(self, x):
		z, y = x.shape
		for _ in range(z):
			for __ in range(y):
				x[_, __] = x[_, __] * (1 - x[_, __])
		return x

	def __tanh_derivative(self, x):
		z, y = x.shape
		for _ in range(z):
			for __ in range(y):
				x[_, __] = 1 - x[_, __]**2
		return x

	def forward_propagate(self, input_matrix):
		self.layers[0].input = self.layers[0].weights * input_matrix + self.layers[0].bias
		self.layers[0].output = self.__sigmoid(self.layers[0].input)
		for i in range(1, self.layers.size):
			self.layers[i].input = self.layers[i].weights * self.layers[i-1].output + self.layers[i].bias
			self.layers[i].output = self.__sigmoid(self.layers[i].input)

	def compute_errors(self, input_matrix, output_matrix):
		self.forward_propagate(input_matrix)
		self.layers[self.layers.size - 1].error = output_matrix - self.layers[self.layers.size - 1].output
		for i in reversed(range(self.layers.size - 1)):
			self.layers[i].error = (np.diagflat(self.__sigmoid_derivative(self.layers[i+1].input)) * self.layers[i+1].weights).T * self.layers[i+1].error

	def back_propagate(self, input_matrix, output_matrix, _lambda, _momentum):
		self.compute_errors(input_matrix, output_matrix)
		self.layers[0].v = 2 * _lambda * (self.layers[0].error * input_matrix.T) + _momentum * self.layers[0].v
		self.layers[0].vb = 2 * _lambda * (self.layers[0].error) + _momentum * self.layers[0].vb
		self.layers[0].weights = self.layers[0].weights + self.layers[0].v
		self.layers[0].bias = self.layers[0].bias + self.layers[0].vb
		for i in range(1, self.layers.size):
			self.layers[i].v = 2 * _lambda * (self.layers[i].error * self.layers[i-1].output.T) + _momentum * self.layers[i].v
			self.layers[i].vb = 2 * _lambda * (self.layers[i].error) + _momentum * self.layers[i].vb
			self.layers[i].weights = self.layers[i].weights + self.layers[i].v
			self.layers[i].bias = self.layers[i].bias + self.layers[i].vb

input_matrix = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).T
arr = np.arange(4)
layer1 = neuronlayer(2, 4)
layer2 = neuronlayer(4, 2)
layerarray = np.array([layer1, layer2])
network = neuralnetwork(layerarray)
_lambda = 0.2
_momentum = 0.6
ex = list()
why = list()
for j in range(10000):
	for x in range(4):
		network.back_propagate(input_matrix[arr[x]].T, input_matrix[arr[x]].T, _lambda, _momentum)
	if j % 10 == 0:
		print("Progress ", 100*j/10000, "%")
		ex.append(j)
		cost = (layer2.error[0]**2 + layer2.error[1]**2 + layer2.error[2]**2 + layer2.error[3]**2)
		why.append(float(cost))
	np.random.shuffle(arr)

for k in range(4):
	network.forward_propagate(input_matrix[k].T)
	print("###########################################")
	print(input_matrix[k].T)
	for q in range(4):
		print(round(float(layer2.output[q,0]),1))
plt.plot(ex, why)
plt.show()
