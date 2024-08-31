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
    def __init__(self, hidden_state_size, mamba_seq_size, reward_size):
        super(NeuralControlCritic, self).__init__()

        assert hidden_state_size % mamba_seq_size == 0, "hidden_state_size must be divisible by mamba_seq_size"

        self.loss = nn.MSELoss()

        self.mamba_seq_size = mamba_seq_size
        self.hidden_size = (hidden_state_size // mamba_seq_size) * 8

        # # Assert that hidden_size is divisible by mamba_seq_size
        # assert hidden_size % mamba_seq_size == 0, "hidden_size must be divisible by mamba_seq_size"

        #self.loss = nn.MSELoss()

        self.mlp_in = MLP(1, hidden_state_size, hidden_state_size*self.mamba_seq_size, hidden_state_size*self.mamba_seq_size)
        self.mamba = nn.Sequential(
            Mamba2(self.hidden_size),
            Mamba2(self.hidden_size),
        )
        #self.pre_mlp_flat = nn.Flatten(start_dim=0)
        self.mlp_out = MLP(1, self.hidden_size*self.mamba_seq_size, self.hidden_size, reward_size)


    def forward(self, x):#, steps_left, current_r):
        batch_dim = x.shape[0]

        # x = x.clone()
        # x[:,:,0] = steps_left
        # x[:,:,1] = current_r


        #pdb.set_trace()
        x = self.mlp_in(x)
        
        x = x.view(batch_dim, self.mamba_seq_size, self.hidden_size)
        x = self.mamba(x)
        x = x.view(batch_dim, self.mamba_seq_size*self.hidden_size)

        x = self.mlp_out(x)

        return x

    # def loss(self, a, b):
    #     return torch.sum(torch.square(a-b))

