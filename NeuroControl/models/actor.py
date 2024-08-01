import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NeuroControl.models.transcnn import TransCNN
from NeuroControl.models.transformer import Transformer
from NeuroControl.models.mlp import MLP
from mamba_ssm import Mamba2



class NeuronControlActor(nn.Module):
    def __init__(self, state_size, hidden_size, action_dims):
        super(NeuronControlActor, self).__init__()
        # action_size is the total multiplied action_dims
        action_size = np.prod(action_dims)
        # action_dims is a tuple of (num_stim_neurons, stim_time_steps)
        
        # Multiplied by two because we need both the mean and std for each action
        # self.out_dim = stim_time_steps * num_stim_neurons * 2
        # self.model = TransCNN(num_input_frames, image_n, dim, self.out_dim, num_trans_layers)
        self.mlp_in = MLP(2, state_size, hidden_size)
        self.mamba = nn.Sequential(
            Mamba2(hidden_size),
            Mamba2(hidden_size),
            Mamba2(hidden_size),
            Mamba2(hidden_size),
        )
        self.mlp_out = MLP(2, hidden_size, action_size)
        self.act = nn.Sigmoid()
        #self.action_dims = (num_stim_neurons, stim_time_steps)

    def forward(self, x):

        x = self.mlp_in(x)
        x = self.mamba(x)
        x = self.mlp_out(x)
        # Reshape the output to match the action dimensions
        x = x.view(-1, *self.action_dims)
        x = self.act(x)

        return x

    def entropy(self, x):

        x = (-x * torch.log(x)) - ((1 - x) * torch.log(1 - x))
        x = torch.sum(x)

        return x