import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba2 as Mamba
from NeuroControl.models.mlp import MLP
from NeuroControl.custom_functions.utils import STMNsampler, symlog, symexp
import pdb



############## SPLIT INTO Z HAT PREDICTOR AND H STATE GENERATOR

class NeuralRecurrentDynamicsModel(nn.Module):
    def __init__(self, hidden_state_size, obs_latent_size, action_size, seq_size, per_image_discrete_latent_size_sqrt):
        super(NeuralRecurrentDynamicsModel, self).__init__()

        # Ensure latent_size is divisible by seq_size
        #assert hidden_state_size % seq_size == 0, "latent_size must be divisible by seq_size"

        # Initialize model parameters
        self.hidden_state_size = hidden_state_size
        self.seq_size = seq_size
        self.per_image_discrete_latent_size_sqrt = per_image_discrete_latent_size_sqrt

        self.obs_latent_size = obs_latent_size

        self.pre_mlp_size = hidden_state_size + obs_latent_size + action_size
        #self.hidden_size = 1024

        self.pre_post_mamba_size = ((2 ** ((int(self.pre_mlp_size) - 1).bit_length() - 1)))

        assert self.pre_post_mamba_size % self.seq_size == 0, "per_item_mamba_size must be divisible by 2"
        self.hidden_mamba_size = (self.pre_post_mamba_size // self.seq_size)

        # Initial MLP layer
        self.mlp_0 = MLP(2, self.pre_mlp_size, self.pre_post_mamba_size, self.pre_post_mamba_size)

        self.pre_gru_mamba = nn.Sequential(
            Mamba(self.hidden_mamba_size),
            #Mamba(self.hidden_mamba_size),
        )
        self.gru = nn.GRUCell(self.pre_post_mamba_size, self.hidden_state_size)

        # Mamba layers for sequence processing
        # Input shape: (batch, seq, dim)
        self.pre_z_pred_mamba_mlp = MLP(2, self.hidden_state_size, self.pre_post_mamba_size, self.pre_post_mamba_size)
        self.z_pred_mamba = nn.Sequential(
            Mamba(self.hidden_mamba_size),
            #Mamba(self.hidden_mamba_size),
            #Mamba(self.hidden_mamba_size),
        )


        # Final MLP layer
        self.mlp_1 = MLP(2, self.pre_post_mamba_size, self.obs_latent_size, self.obs_latent_size)

        self.softmax_act = nn.Softmax(dim=1)
        self.sampler = STMNsampler()
        # self.discretizer = nn.Sequential(
        #     nn.Softmax(dim=1),
        #     STMNsampler(),
        # )

    def forward(self, obs_latent, h_state, action):

        batch_dim = obs_latent.shape[0]
        
        action = torch.flatten(action, start_dim=1)
        obs_latent = torch.flatten(obs_latent, start_dim=1)
        #print(obs_latent.size(), h_state.size(), action.size())
        #pdb.set_trace()
        x = torch.cat((obs_latent, h_state, action), dim=1)
        # Paxss input through initial MLP
        #pdb.set_trace()
        x = self.mlp_0(x)


        gru_x = self.pre_gru_mamba(x.view(batch_dim, self.seq_size, self.hidden_mamba_size)).view(batch_dim, self.pre_post_mamba_size)
        h_state_hat = self.gru(gru_x, h_state)

        #x = 
        x = self.pre_z_pred_mamba_mlp(h_state)

        # Reshape input for Mamba layers
        x = x.view(batch_dim, self.seq_size, self.hidden_mamba_size)

        # Process through Mamba layers
        x = self.z_pred_mamba(x)

        # Reshape output back to original dimensions
        x = x.view(batch_dim, self.pre_post_mamba_size)

        #pdb.set_trace()

        # Pass through final MLP
        x = self.mlp_1(x)

        x = x.view(batch_dim*self.per_image_discrete_latent_size_sqrt * self.seq_size, self.per_image_discrete_latent_size_sqrt)
        distributions = self.softmax_act(x)
        samples = self.sampler(distributions)
        #x = self.discretizer(x)
        obs_latent_sample_hat = samples.view(batch_dim, self.obs_latent_size)
        obs_latent_distribution_hat = distributions.view(batch_dim, self.obs_latent_size)


        return obs_latent_sample_hat, obs_latent_distribution_hat, h_state_hat


class NeuralSeqModel(nn.Module):
    def __init__(self, hidden_state_size, obs_latent_size, action_size, seq_size, per_image_discrete_latent_size_sqrt):
        super(NeuralRecurrentDynamicsModel, self).__init__()

        # Ensure latent_size is divisible by seq_size
        #assert hidden_state_size % seq_size == 0, "latent_size must be divisible by seq_size"

        # Initialize model parameters
        self.hidden_state_size = hidden_state_size
        self.seq_size = seq_size
        self.per_image_discrete_latent_size_sqrt = per_image_discrete_latent_size_sqrt

        self.obs_latent_size = obs_latent_size

        self.pre_mlp_size = hidden_state_size + obs_latent_size + action_size
        #self.hidden_size = 1024

        self.pre_post_mamba_size = ((2 ** ((int(self.pre_mlp_size) - 1).bit_length() - 1)))

        assert self.pre_post_mamba_size % self.seq_size == 0, "per_item_mamba_size must be divisible by 2"
        self.hidden_mamba_size = (self.pre_post_mamba_size // self.seq_size)

        # Initial MLP layer
        self.mlp_0 = MLP(2, self.pre_mlp_size, self.pre_post_mamba_size, self.pre_post_mamba_size)

        self.pre_gru_mamba = nn.Sequential(
            Mamba(self.hidden_mamba_size),
            Mamba(self.hidden_mamba_size),
        )
        self.gru = nn.GRUCell(self.pre_post_mamba_size, self.hidden_state_size)





    def forward(self, obs_latent, h_state, action):

        batch_dim = obs_latent.shape[0]
        
        action = torch.flatten(action, start_dim=1)
        obs_latent = torch.flatten(obs_latent, start_dim=1)

        x = torch.cat((obs_latent, h_state, action), dim=1)


        x = self.mlp_0(x)


        gru_x = self.pre_gru_mamba(x.view(batch_dim, self.seq_size, self.hidden_mamba_size)).view(batch_dim, self.pre_post_mamba_size)
        h_state_hat = self.gru(gru_x, h_state)



        return h_state_hat


class NeuralRepModel(nn.Module):
    def __init__(self, hidden_state_size, obs_latent_size, seq_size, per_image_discrete_latent_size_sqrt):
        super(NeuralRecurrentDynamicsModel, self).__init__()

        # Ensure latent_size is divisible by seq_size
        #assert hidden_state_size % seq_size == 0, "latent_size must be divisible by seq_size"

        # Initialize model parameters
        self.hidden_state_size = hidden_state_size
        self.seq_size = seq_size
        self.per_image_discrete_latent_size_sqrt = per_image_discrete_latent_size_sqrt

        self.obs_latent_size = obs_latent_size

        self.pre_mlp_size = hidden_state_size
        #self.hidden_size = 1024

        self.pre_post_mamba_size = ((2 ** ((int(self.pre_mlp_size) - 1).bit_length() - 1)))

        assert self.pre_post_mamba_size % self.seq_size == 0, "per_item_mamba_size must be divisible by 2"
        self.hidden_mamba_size = (self.pre_post_mamba_size // self.seq_size)


        # Mamba layers for sequence processing
        # Input shape: (batch, seq, dim)
        self.pre_z_pred_mamba_mlp = MLP(2, self.hidden_state_size, self.pre_post_mamba_size, self.pre_post_mamba_size)
        self.z_pred_mamba = nn.Sequential(
            Mamba(self.hidden_mamba_size),
            Mamba(self.hidden_mamba_size),
            #Mamba(self.hidden_mamba_size),
        )


        # Final MLP layer
        self.mlp_1 = MLP(1, self.pre_post_mamba_size, self.obs_latent_size, self.obs_latent_size)

        self.softmax_act = nn.Softmax(dim=1)
        self.sampler = STMNsampler()
        # self.discretizer = nn.Sequential(
        #     nn.Softmax(dim=1),
        #     STMNsampler(),
        # )

    def forward(self, h_state):

        batch_dim = obs_latent.shape[0]

        x = self.pre_z_pred_mamba_mlp(h_state)

        # Reshape input for Mamba layers
        x = x.view(batch_dim, self.seq_size, self.hidden_mamba_size)

        # Process through Mamba layers
        x = self.z_pred_mamba(x)

        # Reshape output back to original dimensions
        x = x.view(batch_dim, self.pre_post_mamba_size)

        #pdb.set_trace()

        # Pass through final MLP
        x = self.mlp_1(x)

        x = x.view(batch_dim*self.per_image_discrete_latent_size_sqrt * self.seq_size, self.per_image_discrete_latent_size_sqrt)
        distributions = self.softmax_act(x)
        samples = self.sampler(distributions)
        #x = self.discretizer(x)
        obs_latent_sample_hat = samples.view(batch_dim, self.obs_latent_size)
        obs_latent_distribution_hat = distributions.view(batch_dim, self.obs_latent_size)


        return obs_latent_sample_hat, obs_latent_distribution_hat

