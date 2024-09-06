import torch
from torch import nn
from torch.nn import functional as F
from NeuroControl.custom_functions.utils import RMSNorm

class MLP(nn.Module):
    def __init__(self, layers_num, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()

        # Add input layer
        self.layers.add_module('input_layer', nn.Linear(input_size, hidden_size))
        # Add activation function after input layer
        self.layers.add_module('layer norm_0', RMSNorm(hidden_size))
        # Add activation function after input layer
        self.layers.add_module('activation_0', nn.GELU())

        # Add hidden layers
        for i in range(1, layers_num):
            self.layers.add_module(f'hidden_layer_{i}', nn.Linear(hidden_size, hidden_size))

            self.layers.add_module(f'layer norm_{i}', RMSNorm(hidden_size))

            # Add activation function after each hidden layer
            self.layers.add_module(f'activation_{i}', nn.GELU())

        # Add output layer
        self.layers.add_module('output_layer', nn.Linear(hidden_size, output_size))
        
        
    def forward(self, x):
        return self.layers(x)