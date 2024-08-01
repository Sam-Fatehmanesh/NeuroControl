import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NeuroControl.models.transcnn import TransCNN
from NeuroControl.models.mlp import MLP
from NeuroControl.models.encoder import SpikePositionEncoding
from mamba_ssm import Mamba2


class NeuralControlCritic(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(NeuralControlCritic, self).__init__()


        self.loss = nn.MSELoss()

        self.mlp_in = MLP(2, state_size + 2, hidden_size)
        self.mamba = nn.Sequential(
            Mamba2(hidden_size),
            Mamba2(hidden_size),
            Mamba2(hidden_size),
            Mamba2(hidden_size),
        )
        self.mlp_out = MLP(2, hidden_size, 1)


    def forward(self, x, steps_left, current_r):
        
        x = torch.cat((x, steps_left, current_r), dim=1)
        x = self.mlp_in(x)
        x = self.mamba(x)
        x = self.mlp_out(x)

        return x

