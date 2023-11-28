# Modules
import torch
import numpy as np

# Function imported
from torch import optim

# Training algorithm for the Neural Network

def trainingNN(MyNN, x_train, y_train, epoch):
 
 # The input data
 idata = x_train

 # The output data (target)
 odata = y_train

 # Define the Binary Cross - Entropy Loss function
 loss = torch.nn.BCELoss()

 # Define the optimizer
 optimizer = optim.SGD(MyNN.parameters(), lr=1e-1)

 print('\nStart training:')

 for i in range(epoch):

  # Calculate the output of the Neural Network from the given input dataset
  ynn = MyNN(idata)

  # Calculate the error between the target and the output
  error = loss(ynn, odata)

  # Calculate the gradient for each tensor of weight and bias
  error.backward()  
  
  # Update the parameters
  optimizer.step()
  optimizer.zero_grad()

  # Print the error every 500 iterations
  if (i == 0):
   print("\n","The error (every 500 steps):")
  if np.mod(i, 500) == 0:
   print(error)

 return print('\nTraining has finished.\n')


