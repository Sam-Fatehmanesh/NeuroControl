import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NeuroControl.models.mlp import MLP
from mamba_ssm import Mamba2 as Mamba
from NeuroControl.custom_functions.utils import STMNsampler
import pdb



class NeuralControlActor(nn.Module):
    def __init__(self, hidden_state_and_image_lat_size, hidden_size, action_dims):
        super(NeuralControlActor, self).__init__()

        self.hidden_size = hidden_size
        self.action_time_dim_size = action_dims[0]

        # Assert that hidden_size is divisible by saction_time_dim_size
        assert hidden_size % self.action_time_dim_size == 0, "hidden_size must be divisible by action_time_dim_size"

        self.seq_embed_size = 8*((((2 ** ((int(hidden_state_and_image_lat_size) - 1).bit_length()))) // self.action_time_dim_size) // 8)
        #pdb.set_trace()
        self.post_pre_mamba_size = self.seq_embed_size*self.action_time_dim_size

        # action_size is the total multiplied action_dims
        self.action_size = np.prod(action_dims)
        self.action_dims = action_dims
        
        # action_dims is a tuple of (num_stim_neurons, stim_time_steps)
        
        # Multiplied by two because we need both the mean and std for each action
        # self.out_dim = stim_time_steps * num_stim_neurons * 2
        # self.model = TransCNN(num_input_frames, image_n, dim, self.out_dim, num_trans_layers)
        self.mlp_in = MLP(2, hidden_state_and_image_lat_size, hidden_state_and_image_lat_size // 2, self.post_pre_mamba_size)
        self.mamba = nn.Sequential(
            nn.Identity(),
            #Mamba(self.seq_embed_size),
            #Mamba(self.seq_embed_size),
            #Mamba(self.hidden_size),
            #Mamba(self.hidden_size),
        )

        self.mlp_out = MLP(2, self.post_pre_mamba_size, hidden_size, self.action_size)
        
        
        self.act = nn.Softmax(dim=2)
        self.sampler = STMNsampler()
        #self.action_dims = (num_stim_neurons, stim_time_steps)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.mlp_in(x)

        x = x.view(batch_size, self.action_time_dim_size, self.seq_embed_size)
        x = self.mamba(x)
        x = x.view(batch_size, self.post_pre_mamba_size)

        x = self.mlp_out(x)

        x = x.view(batch_size, *self.action_dims)
        
        x = self.act(x)
        x = (0.99*x) + (0.01*torch.ones_like(x)/self.action_dims[1])

        dist = x

        x = x.view(batch_size * self.action_time_dim_size, self.action_dims[1])
        sample = self.sampler(x)
        sample = sample.view(batch_size, *self.action_dims)

        return sample, dist


    def entropy(self, dist):
        batch_size = dist.shape[0]
        
        return -torch.sum(dist * torch.log(dist)) / (self.action_time_dim_size * batch_size)




# class NeuralControlActorEnsemble(nn.Module):
#     def __init__(self, state_size, hidden_size, action_dims):
#         super(NeuralControlActorEnsemble, self).__init__()

#         self.hidden_size = hidden_size
#         self.state_size = state_size
#         self.action_dims = action_dims



#         #self.action_dims = (num_stim_neurons, stim_time_steps)

#     def forward(self, x):

        
#         #x = F.log_softmax(self.mlp_out(x), dim=-1)
#         return x.view(*self.action_dims)
