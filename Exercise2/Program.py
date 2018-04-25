#!/usr/bin/python3
import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from MLP import *
from Parser import *
##############
#Parser block#
##############
P = Parser()
P.parse()
_epochs = P.get_epochs()
_lambda = P.get_lambda()
_momentum = P.get_momentum()
_data = P.get_dataset()
_hidden = P.get_hidden()
###########
#CSV block#
###########
df = pd.read_csv(_data + '.data.txt', header = None)
df_height, df_width = df.shape
for i in range(df_height):
    for j in range(df_width):
        if type(df.iloc[i, j]) is str:
            df.iloc[i, j] = ord(df.iloc[i, j])
print(df.iloc[:,df_width - 1])
#############
#Layer block#
#############
layer_array = np.array([])
for i in range(_hidden[0]):
    if i == 0:
        np.append(layer_array, neuronlayer(_hidden[1], df_width - 1))
    else:
        np.append(layer_array, neuronlayer(hidden[i+1], _hidden[i]))
np.append(layer_array, neuronlayer(1, _hidden[len(_hidden) - 1]))
network = neuralnetwork(layer_array)
############################
#Shitty stuff, smells funny#
############################
ox = list()
oy = list()
for j in tqdm(range(_epochs)):
	for x in range(4):
		network.back_propagate(input_matrix[arr[x]].T, output_matrix[arr[x]].T, _lambda, _momentum)
	if j % 1 == 0:
		ex.append(j)
		cost = (layer2.error[0]**2)
		why.append(float(cost))
	np.random.shuffle(arr)

for k in range(4):
	network.forward_propagate(input_matrix[k].T)
	print("###########################################")
	print(input_matrix[k].T)
	print(10*round(float(layer2.output[0]),3))
print(layer2.error[0]**2)
plt.plot(ex, why)
plt.show()
