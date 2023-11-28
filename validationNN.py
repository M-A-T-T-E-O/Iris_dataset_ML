import matplotlib.pyplot as plt
import numpy as np
import torch as nn
import pandas as pd

# Validation algorithm for the Neural Network

def validationNN(MyNN, x_test, y_test):

 # Calculate the output from the trained NN
 output_from_NN = MyNN(x_test)

 # Map to class of flowers to make a comparison with the target
 output_from_NN = output_from_NN.detach().numpy()
 output_from_NN = nn.tensor(np.argmax(output_from_NN, axis = 1))
 y_test = np.argmax(y_test,axis = 1)

 # Plot graphs to compare predictions with target
 col = {'target': y_test, 'predictions': output_from_NN}
 predictions_target = pd.DataFrame(data = col)
 hist = pd.DataFrame(predictions_target, columns=["target", "predictions"])
 hist.plot.bar()
 plt.show()

 # Show both the target and the output from the trained NN
 print("\nThe output of the NN (predictions) is:\n", output_from_NN, "\n\n",
       'The target output (from measurements) is:',
       "\n", y_test, "\n")


