import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NeuroControl.models.transcnn import TransCNN
from NeuroControl.models.mlp import MLP
from NeuroControl.models.encoder import SpikePositionEncoding
from mamba_ssm import Mamba2
import pdb


class NeuralControlCritic(nn.Module):
    def __init__(self, state_size, mamba_seq_size, hidden_size):
        super(NeuralControlCritic, self).__init__()

        self.mamba_seq_size = mamba_seq_size
        self.hidden_size = hidden_size

        # # Assert that hidden_size is divisible by mamba_seq_size
        # assert hidden_size % mamba_seq_size == 0, "hidden_size must be divisible by mamba_seq_size"

        #self.loss = nn.MSELoss()

        self.mlp_in = MLP(2, state_size + 2, hidden_size, hidden_size*self.mamba_seq_size)
        self.mamba = nn.Sequential(
            Mamba2(hidden_size),
            Mamba2(hidden_size),
            Mamba2(hidden_size),
            Mamba2(hidden_size),
        )
        self.pre_mlp_flat = nn.Flatten()
        self.mlp_out = MLP(4, hidden_size*self.mamba_seq_size, hidden_size, 1)


    def forward(self, x, steps_left, current_r):
        #pdb.set_trace()
        
        steps_left = torch.unsqueeze(steps_left, dim=0)
        current_r = torch.unsqueeze(current_r, dim=0)

        x = torch.cat((x, steps_left, current_r), dim=1)

        x = self.mlp_in(x)

        x = x.view(1, self.mamba_seq_size, self.hidden_size)
        x = self.mamba(x)

        x = self.pre_mlp_flat(x)
        x = self.mlp_out(x)

        return x

    def loss(self, a, b):
        return torch.sum(torch.square(a-b))

