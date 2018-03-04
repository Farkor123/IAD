import numpy as np
from scipy import stats as st
import pandas as pd
import matplotlib.pyplot as plt

def stat_summary(data):
    summary = pd.DataFrame(columns=['sepal width', 'sepal length', 'petal width', 'petal length'])
    summary.loc['min'] = [x for x in data.min(numeric_only=True)]
    summary.loc['max'] = [x for x in data.max(numeric_only=True)]
    summary.loc['range'] = [x for x in summary.loc['max']-summary.loc['min']]

    summary.loc['1st quartile'] = [x for x in data.quantile(0.25, numeric_only=True)]
    summary.loc['median'] = [x for x in data.median(numeric_only=True)]
    summary.loc['3rd quartile'] = [x for x in data.quantile(0.75, numeric_only=True)]

    summary.loc['harmonic mean'] = st.hmean(data.iloc[:, 0:4])
    summary.loc['geometric mean'] = st.gmean(data.iloc[:, 0:4])
    summary.loc['gen. mean, p=2'] = [x for x in (( (data.iloc[: ,0:4]**2).sum() / data.shape[0] ) ** (1./2))]
    summary.loc['gen. mean, p=3'] = [x for x in (( (data.iloc[: ,0:4]**3).sum() / data.shape[0] ) ** (1./3))]
    summary.loc['arithmetic mean'] = [x for x in data.mean()]

    summary.loc['variance'] = [x for x in data.var()]
    summary.loc['std deviation'] = [x for x in data.std()]
    summary.loc['kurtosis'] = st.kurtosis(data.iloc[:, 0:4], fisher=False)
    
    return summary

data = pd.read_csv('iris.data', header=None)

setosa = data.loc[data.iloc[:, 4] == 'Iris-setosa']
vcolor = data.loc[data.iloc[:, 4] == 'Iris-versicolor']
virgin = data.loc[data.iloc[:, 4] == 'Iris-virginica']

plt.figure(1)
plt.plot(setosa.iloc[:, 0], setosa.iloc[:, 1], 'r.', label='Iris Setosa')
plt.plot(vcolor.iloc[:, 0], vcolor.iloc[:, 1], 'g.', label='Iris Versicolor')
plt.plot(virgin.iloc[:, 0], virgin.iloc[:, 1], 'b.', label='Iris Virginica')
plt.xlabel('sepal width')
plt.ylabel('sepal length')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(setosa.iloc[:, 2], setosa.iloc[:, 3], 'r.', label='Iris Setosa')
plt.plot(vcolor.iloc[:, 2], vcolor.iloc[:, 3], 'g.', label='Iris Versicolor')
plt.plot(virgin.iloc[:, 2], virgin.iloc[:, 3], 'b.', label='Iris Virginica')
plt.xlabel('petal width')
plt.ylabel('petal length')
plt.legend(loc='lower right')
plt.show()

print('Overall summary:')
print(stat_summary(data))
print('\nIris Setosa summary:')
print(stat_summary(setosa))
print('\nIris Versicolor summary:')
print(stat_summary(vcolor))
print('\nIris Virginica summary:')
print(stat_summary(virgin))
