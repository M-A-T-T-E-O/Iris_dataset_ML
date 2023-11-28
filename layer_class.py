import torch.nn as nn

class PyTorchNN(nn.Module):

    # constructor
    def __init__(self):
        """
        Assigning Linear Layers to class members variables
        """
        super(PyTorchNN, self).__init__()
        self.layer1 = nn.Linear(in_features=4, out_features=3)
        self.layer2 = nn.Linear(in_features=3,  out_features=3)

    # predictor
    def forward(self, x):
        """
        Append Layers
        """
        y = nn.Sequential(self.layer1,
                          self.layer2,
                          nn.Softmax( dim = 1 ))(x)
        return y