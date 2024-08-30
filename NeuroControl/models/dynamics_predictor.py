import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba2 as Mamba
from NeuroControl.models.mlp import MLP

class NeuralRecurrentDynamicsModel(nn.Module):
    def __init__(self, hidden_state_size, obs_latent_size, action_size, seq_size):
        super(NeuralLatentPredictor, self).__init__()

        # Ensure latent_size is divisible by seq_size
        #assert hidden_state_size % seq_size == 0, "latent_size must be divisible by seq_size"

        # Initialize model parameters
        self.hidden_state_size = hidden_state_size
        self.seq_size = seq_size

        self.obs_latent_size = obs_latent_size

        self.pre_mlp_size = hidden_state_size + obs_latent_size + action_size
        #self.hidden_size = 1024

        self.pre_post_mamba_size = (2 ** (self.pre_mlp_size - 1).bit_length()) // 2

        assert self.per_item_mamba_size % self.seq_size == 0, "per_item_mamba_size must be divisible by 2"
        self.hidden_mamba_size = self.pre_post_mamba_size // self.seq_size

        # Initial MLP layer
        self.mlp_0 = MLP(4, self.pre_mlp_size, self.pre_post_mamba_size, self.pre_post_mamba_size)

        self.gru = nn.GRUCell(self.pre_post_mamba_size, self.hidden_state_size)

        # Mamba layers for sequence processing
        # Input shape: (batch, seq, dim)
        self.mamba = nn.Sequential(
            Mamba(self.hidden_mamba_size),
            Mamba(self.hidden_mamba_size),
            Mamba(self.hidden_mamba_size),
            #Mamba(self.hidden_mamba_size),
        )


        


        # Final MLP layer
        self.mlp_1 = MLP(1, self.hidden_mamba_size, self.obs_latent_size, self.obs_latent_size)

    def forward(self, obs_latent, h_state, action):

        batch_dim = obs_latent.shape[0]
        
        action = torch.flatten(action)
        x = torch.cat((obs_latent, h_state, action), dim=1)
        # Pass input through initial MLP
        x = self.mlp_0(x)

        h_state_hat = self.gru(x, h_state)

        # Reshape input for Mamba layers
        x = x.view(batch_dim, self.seq_size, self.hidden_mamba_size)

        # Process through Mamba layers
        x = self.mamba(x)

        # Reshape output back to original dimensions
        x = x.view(batch_dim, self.pre_post_mamba_size)

        # Pass through final MLP
        obs_latent_hat = self.mlp_1(x)

        return obs_latent_hat, h_state_hat

    def loss(self, x, y):
        # Compute MSE loss between predictions and targets
        y_pred = self.forward(x)
        return F.mse_loss(y_pred, y)
    
    def train_step(self, x, y, optimizer):
        # Perform a single training step
        loss = self.loss(x, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
