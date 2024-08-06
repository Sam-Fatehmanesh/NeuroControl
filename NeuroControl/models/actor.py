import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NeuroControl.models.transcnn import TransCNN
from NeuroControl.models.transformer import Transformer
from NeuroControl.models.mlp import MLP
from mamba_ssm import Mamba2
import pdb



class NeuralControlActor(nn.Module):
    def __init__(self, state_size, hidden_size, action_dims):
        super(NeuralControlActor, self).__init__()

        self.hidden_size = hidden_size
        self.action_time_dim_size = action_dims[1]
        self.seq_embed_size = int(hidden_size / self.action_time_dim_size)
        # Assert that hidden_size is divisible by action_time_dim_size
        assert hidden_size % self.action_time_dim_size == 0, "hidden_size must be divisible by action_time_dim_size"

        # action_size is the total multiplied action_dims
        action_size = np.prod(action_dims)
        self.action_dims = action_dims
        
        # action_dims is a tuple of (num_stim_neurons, stim_time_steps)
        
        # Multiplied by two because we need both the mean and std for each action
        # self.out_dim = stim_time_steps * num_stim_neurons * 2
        # self.model = TransCNN(num_input_frames, image_n, dim, self.out_dim, num_trans_layers)
        self.mlp_in = MLP(2, state_size, hidden_size, self.action_time_dim_size*hidden_size)
        self.mamba = nn.Sequential(
            Mamba2(self.hidden_size),
            Mamba2(self.hidden_size),
            Mamba2(self.hidden_size),
            Mamba2(self.hidden_size),
        )
        self.flat_premlp = nn.Flatten()
        self.mlp_out = MLP(2, self.action_time_dim_size*hidden_size, hidden_size, action_size)
        
        
        self.act = nn.LogSigmoid()
        #self.action_dims = (num_stim_neurons, stim_time_steps)

    def forward(self, x):
        x = self.mlp_in(x)
        x = x.view(1, self.action_time_dim_size, self.hidden_size)
        x = self.mamba(x)
        x = self.flat_premlp(x)
        x = self.mlp_out(x)
        x = self.act(x)
        #x = F.log_softmax(self.mlp_out(x), dim=-1)
        return x.view(*self.action_dims)


    def entropy(self, x):
        x = torch.exp(x)  # Convert logits to probabilities
        eps = 1e-7  # Small epsilon to prevent log(0)
        x = torch.clamp(x, eps, 1 - eps)  # Clip values to be between eps and 1-eps
        entropy = (-x * torch.log(x)) - ((1 - x) * torch.log(1 - x))
        return torch.sum(entropy)

    # def entropy(self, x):
    #     probs = torch.exp(x)
    #     return -torch.sum(probs * x + (1 - probs) * torch.log1p(-probs.exp()))
