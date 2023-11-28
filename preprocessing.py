# Modules
import torch as nn
import numpy as np

# Function imported
from sklearn.model_selection import train_test_split

# Dataset preprocessing

def preprocessing(dataset):
 
 # Deletes rows containing nan or empty values
 dataset.replace('', np.nan, inplace=True)
 dataset.dropna(inplace=True)

 # Codify iris classes (Iris-setosa = 0, Iris-versicolor = 1, Iris-virginica = 2)
 dataset['Class_ID'] = dataset['Class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

 # Split dataset based on Iris class
 setosa = dataset[dataset['Class_ID'] == 0]
 versicolor = dataset[dataset['Class_ID'] == 1]
 virginica = dataset[dataset['Class_ID'] == 2]

 # Get the input and the output (target)
 setosa_x = setosa[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
 versicolor_x = versicolor[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
 virginica_x = virginica[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
 setosa_y = setosa[["Class_ID"]].values
 versicolor_y = versicolor[["Class_ID"]].values
 virginica_y = virginica[["Class_ID"]].values

 # Split data into training set and test set
 setosa_x_train, setosa_x_test, setosa_y_train, setosa_y_test = train_test_split(setosa_x, setosa_y, test_size=0.3, shuffle = False)
 versicolor_x_train, versicolor_x_test, versicolor_y_train, versicolor_y_test = train_test_split(versicolor_x, versicolor_y, test_size=0.3, shuffle = False)
 virginica_x_train, virginica_x_test, virginica_y_train, virginica_y_test = train_test_split(virginica_x, virginica_y, test_size=0.3, shuffle = False)

 # Creation of the data training set
 x_train = np.concatenate([setosa_x_train, versicolor_x_train, virginica_x_train])
 y_train = np.concatenate([setosa_y_train, versicolor_y_train, virginica_y_train])
 x_train = nn.tensor(x_train).float()
 y_train = nn.tensor(y_train).float()

 # Creation of the data test set
 x_test = np.concatenate([setosa_x_test, versicolor_x_test, virginica_x_test])
 y_test = np.concatenate([setosa_y_test, versicolor_y_test, virginica_y_test])
 x_test = nn.tensor(x_test).float()
 y_test = nn.tensor(y_test).float()

 # Map both training and test set into binary matrix
 temp = nn.empty(y_train.shape[0], 3)
 temp[:,0] = y_train[:,0] == 0
 temp[:,1] = y_train[:,0] == 1
 temp[:,2] = y_train[:,0] == 2
 y_train = temp
 temp = nn.empty(y_test.shape[0], 3)
 temp[:,0] = y_test[:,0] == 0
 temp[:,1] = y_test[:,0] == 1
 temp[:,2] = y_test[:,0] == 2
 y_test = temp

 return x_train, y_train, x_test, y_test


