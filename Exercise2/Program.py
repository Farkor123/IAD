#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Parser
import MLP
import FUNCTIONS
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
_mode = P.get_mode()
_path = P.get_path()
_mistake = 3
#############
# CSV block #
#############
if _data == 'abalone':
    if _mode == 1:
        _data += '_test'
    else:
        _data += '_train'
    input_matrix, output_matrix, temp_matrix, df_width, df_height = FUNCTIONS.createabalonset(_data)
if _data == 'mnist':
    if _mode == 1:
        _data += '_test'
    else:
        _data += '_train'
    input_matrix, output_matrix, temp_matrix, df_width, df_height = FUNCTIONS.createabalonset(_data)
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
if _mode == 0:
    FUNCTIONS.learn(_epochs, df_height, df_width, _lambda, _momentum, temp_matrix, input_matrix, output_matrix, layer_array, network, _path)
else:
    FUNCTIONS.test(df_height, df_width, temp_matrix, input_matrix, output_matrix, _path, _mistake)
