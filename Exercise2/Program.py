#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Parser
import MLP
import ABALON
################
# Parser block #
################
np.seterr(over = 'ignore')
P = Parser.Parser()
P.parse()
_epochs = P.get_epochs()
_lambda = P.get_lambda()
_momentum = P.get_momentum()
_data = P.get_dataset()
_hidden = P.get_hidden()
_activation_functions = P.get_activation_functions()
#############
# CSV block #
#############
if _data == 'abalone':
    input_matrix, output_matrix, temp_matrix, df_width, df_height = ABALON.createdataset(_data)
###############
# Layer block #
###############
layer_array = np.array([])
for i in range(_hidden[0]):
    if i == 0:
        layer_array = np.append(layer_array, np.array(
            [MLP.NeuronLayer(_hidden[1], df_width - 1, _activation_functions[0])]))
    else:
        layer_array = np.append(
            layer_array, MLP.NeuronLayer(_hidden[i + 1], _hidden[i], _activation_functions[i]))
layer_array = np.append(
    layer_array, MLP.NeuronLayer(temp_matrix.size, _hidden[len(_hidden) - 1], _activation_functions[-1]))
network = MLP.NeuralNetwork(layer_array)
##############################
# Shitty stuff, smells funny #
##############################
ox = list()
oy = list()
arr = list(range(0, df_height))
plt.ion()
fig = plt.figure()
highest = 0
lowest = 99999
for j in range(_epochs):
    cost = 0
    for x in range(df_height):
        network.back_propagate(
            input_matrix[arr[x]].T, output_matrix[arr[x]].T, _lambda, _momentum)
        for q in range(temp_matrix.size):
            cost += (layer_array[-1].error[q, 0] * layer_array[-1].error[q, 0])
    np.random.shuffle(arr)
    _lambda *= 0.98
    perc = 0
    for k in range(df_height):
        network.forward_propagate(input_matrix[k].T)
        if np.matrix.sum(output_matrix[k]) == np.matrix.sum(layer_array[-1].output[:] > 0.45):
            perc += 1
    if 100 * perc / df_height > highest:
        highest = round(100 * perc / df_height, 2)
    if round(float(cost), 2) < lowest:
        lowest = round(float(cost), 2)
    print(str(round(100 * perc / df_height, 2)) + "% (highest accuracy: " + str(highest) + "%, lowest error: " + str(lowest) + ")")
    ox.append(j)
    oy.append(float(cost))
    plt.plot(ox, oy)
    plt.show()
    plt.pause(0.0001)

cost = 0
for q in range(temp_matrix.size):
    cost = cost + (layer_array[-1].error[q, 0]**2)
print(cost)
plt.plot(ox, oy)
plt.show()
