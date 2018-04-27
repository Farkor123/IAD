#!/usr/bin/python3
import pandas as pd
import numpy as np

df = pd.read_csv('abalone' + '.data', header=None)
df_height, df_width = df.shape
for i in range(df_height):
    for j in range(df_width):
        if type(df.iloc[i, j]) is str:
            df.iloc[i, j] = ord(df.iloc[i, j])

dupa = df.iloc[:, 8].as_matrix()
dupa = pd.unique(dupa)
print(dupa)
df = np.asmatrix(df.as_matrix())
dupa = np.sort(dupa)
print(dupa)

classes = pd.DataFrame(np.zeros([df_height, dupa.size]))
for i in range(dupa.size):
    classes.iloc[:, i] = df[:, -1] == dupa[i]
print(classes.columns)
