import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle
import MLP

def createmnistset(_data):
    df = pd.read_csv(_data + '.csv', header=None)
    df_height, df_width = df.shape
    inputs = np.asmatrix(df.as_matrix())
    input_matrix = inputs[:, 1:df_width]
    temp_matrix = df.iloc[:, 0].as_matrix()
    temp_matrix = pd.unique(temp_matrix)
    df = np.asmatrix(df.as_matrix())
    temp_matrix = np.sort(temp_matrix)
    classes = pd.DataFrame(np.zeros([df_height, temp_matrix.size]))
    for i in range(temp_matrix.size):
        classes.iloc[:, i] = df[:, 0] >= temp_matrix[i]
    classes = np.asmatrix(classes.as_matrix())
    output_matrix = classes.astype(np.float32)
    return input_matrix, output_matrix, temp_matrix, df_width, df_height

def createabalonset(_data):
    df = pd.read_csv(_data + '.csv', header=None)
    df_height, df_width = df.shape
    inputs = np.asmatrix(df.as_matrix())
    input_matrix = inputs[:, :df_width - 1]
    temp_matrix = df.iloc[:, -1].as_matrix()
    temp_matrix = pd.unique(temp_matrix)
    df = np.asmatrix(df.as_matrix())
    temp_matrix = np.sort(temp_matrix)
    classes = pd.DataFrame(np.zeros([df_height, temp_matrix.size]))
    for i in range(temp_matrix.size):
        classes.iloc[:, i] = df[:, -1] >= temp_matrix[i]
    classes = np.asmatrix(classes.as_matrix())
    output_matrix = classes.astype(np.float32)
    return input_matrix, output_matrix, temp_matrix, df_width, df_height

def learn(_epochs, df_height, df_width, _lambda, _momentum, temp_matrix, input_matrix, output_matrix, layer_array, network, _path):
    #ox = list()
    #oy = list()
    arr = list(range(0, df_height))
    #plt.ion()
    #fig = plt.figure()
    for j in tqdm(range(_epochs)):
        #cost = 0
        for x in tqdm(range(df_height)):
            network.back_propagate(
                input_matrix[arr[x]].T, output_matrix[arr[x]].T, _lambda, _momentum)
            #for q in range(temp_matrix.size):
                #cost += (layer_array[-1].error[q, 0] * layer_array[-1].error[q, 0])
        np.random.shuffle(arr)
        _lambda *= 0.98
        #ox.append(j)
        #oy.append(float(cost))
        #plt.plot(ox, oy)
        #plt.show()
        #plt.pause(0.0001)
        pickle.dump(network, open(_path, 'wb'))

def test(df_height, df_width, temp_matrix, input_matrix, output_matrix, _path, _mistake):
    network = pickle.load(open(_path, 'rb'))
    layer_array = network.layers
    perc = 0
    for k in range(df_height):
        network.forward_propagate(input_matrix[k].T)
        if np.abs(np.matrix.sum(output_matrix[k]) - np.matrix.sum(layer_array[-1].output[:] > 0.45)) <= _mistake:
            perc += 1
    print(str(round(100 * perc / df_height, 2)))
