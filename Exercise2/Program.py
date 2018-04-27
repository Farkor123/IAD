#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import Parser
import sys
import MLP
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
#############
# CSV block #
#############
df = pd.read_csv(_data + '.data.txt', header=None)
df_height, df_width = df.shape
for i in range(df_height):
    for j in range(df_width):
        if type(df.iloc[i, j]) is str:
            df.iloc[i, j] = ord(df.iloc[i, j])
inputs = np.asmatrix(df.as_matrix())
input_matrix = inputs[:, :df_width - 1]
dupa = df.iloc[:, 8].as_matrix()
dupa = pd.unique(dupa)
df = np.asmatrix(df.as_matrix())
dupa = np.sort(dupa)
classes = pd.DataFrame(np.zeros([df_height, dupa.size]))
for i in range(dupa.size):
    classes.iloc[:, i] = df[:, -1] == dupa[i]
classes = np.asmatrix(classes.as_matrix())
output_matrix = classes
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
    layer_array, MLP.neuronlayer(dupa.size, _hidden[len(_hidden) - 1]))
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
            input_matrix[arr[x]].T, output_matrix[arr[x]].T, _lambda, _momentum)
    if j % 1 == 0:
        ox.append(j)
        cost = 0
        for q in range(dupa.size):
            cost = cost + (layer_array[-1].error[q, 0]**2)
        oy.append(float(cost))
    np.random.shuffle(arr)

for k in range(4):
    network.forward_propagate(input_matrix[k].T)
    print("\n###########################################")
    print(output_matrix[k].T, layer_array[-1].output)
cost = 0
for q in range(dupa.size):
    cost = cost + (layer_array[-1].error[q, 0]**2)
print(cost)
plt.plot(ox, oy)
plt.show()
