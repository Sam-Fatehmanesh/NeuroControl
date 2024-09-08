import torch
from torch import nn
from torch.nn import functional as F
from NeuroControl.custom_functions.utils import RMSNorm

class MLP(nn.Module):
    def __init__(self, layers_num, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_norm = RMSNorm(hidden_size)
        self.input_activation = nn.GELU()

        self.hidden_layers = nn.ModuleList()
        for _ in range(layers_num - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                RMSNorm(hidden_size),
                nn.GELU()
            )
            self.hidden_layers.append(layer)

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        residual = x
        x = self.input_norm(x)
        x = self.input_activation(x)

        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            x = residual + layer(x)
            residual = x


        # Output layer
        x = self.output_layer(x)

        return x
