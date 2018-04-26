#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import Parser
import MLP
################
# Parser block #
################
P = Parser.Parser()
P.parse()
_epochs = P.get_epochs()
_lambda = P.get_lambda()
_momentum = P.get_momentum()
_data = P.get_dataset()
_hidden = P.get_hidden()
#############
# CSV block #
#############
df = pd.read_csv(_data + '.data', header=None)
df_height, df_width = df.shape
for i in range(df_height):
    for j in range(df_width):
        if type(df.iloc[i, j]) is str:
            df.iloc[i, j] = ord(df.iloc[i, j])
df = np.asmatrix(df.as_matrix())
input_matrix = df[:, :df_width - 1]
output_matrix = df[:, df_width - 1]
output_matrix = output_matrix / 100
###############
# Layer block #
###############
layer_array = np.array([])
for i in range(_hidden[0]):
    if i == 0:
        layer_array = np.append(layer_array, np.array(
            [MLP.neuronlayer(_hidden[1], df_width - 1)]))
    else:
        layer_array = np.append(
            layer_array, MLP.neuronlayer(_hidden[i + 1], _hidden[i]))
layer_array = np.append(
    layer_array, MLP.neuronlayer(1, _hidden[len(_hidden) - 1]))
network = MLP.neuralnetwork(layer_array)
##############################
# Shitty stuff, smells funny #
##############################
ox = list()
oy = list()
arr = list(range(0, df_height))
for j in tqdm(range(_epochs)):
    for x in tqdm(range(df_height)):
        network.back_propagate(
            input_matrix[arr[x]].T, output_matrix[arr[x]], _lambda, _momentum)
    if j % 1 == 0:
        ox.append(j)
        cost = (layer_array[layer_array.size - 1].error[0]**2)
        oy.append(float(cost))
    np.random.shuffle(arr)

for k in range(4):
    network.forward_propagate(input_matrix[k].T)
    print("###########################################")
    print(100 * output_matrix[k])
    print(100 * float(network.layers[layer_array.size - 1].output))
print(layer_array[layer_array.size - 1].error[0]**2)
plt.plot(ox, oy)
plt.show()
