import pandas as pd
import numpy as np


def createdataset(_data):
    df = pd.read_csv(_data + '.data', header=None)
    df_height, df_width = df.shape
    for i in range(df_height):
        for j in range(df_width):
            if type(df.iloc[i, j]) is str:
                df.iloc[i, j] = ord(df.iloc[i, j]) / 100
    inputs = np.asmatrix(df.as_matrix())
    input_matrix = inputs[:, :df_width - 1]
    temp_matrix = df.iloc[:, 8].as_matrix()
    temp_matrix = pd.unique(temp_matrix)
    df = np.asmatrix(df.as_matrix())
    temp_matrix = np.sort(temp_matrix)
    classes = pd.DataFrame(np.zeros([df_height, temp_matrix.size]))
    for i in range(temp_matrix.size):
        classes.iloc[:, i] = df[:, -1] >= temp_matrix[i]
    classes = np.asmatrix(classes.as_matrix())
    output_matrix = classes.astype(np.float32)
    return input_matrix, output_matrix, temp_matrix, df_width, df_height
