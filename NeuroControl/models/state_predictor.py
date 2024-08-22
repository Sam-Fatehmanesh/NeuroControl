import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba2 as Mamba
from NeuroControl.models.mlp import MLP

class NeuralStatePredictor(nn.Module):
    def __init__(self, latent_size, seq_size, action_size, device):
        super(NeuralLatentPredictor, self).__init__()

        # Ensure latent_size is divisible by seq_size
        assert latent_size % seq_size == 0, "latent_size must be divisible by seq_size"

        # Initialize model parameters
        self.device = device
        self.latent_size = latent_size
        self.seq_size = seq_size
        self.hidden_size = latent_size // seq_size

        # Initial MLP layer
        self.mlp_0 = MLP(2, latent_size+action_size, latent_size+action_size, latent_size)

        # Mamba layers for sequence processing
        # Input shape: (batch, seq, dim)
        self.mamba = nn.Sequential(
            Mamba(self.hidden_size),
            Mamba(self.hidden_size),
            Mamba(self.hidden_size),
            Mamba(self.hidden_size),
        )

        # Final MLP layer
        self.mlp_1 = MLP(2, latent_size, latent_size, latent_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        # Pass input through initial MLP
        x = self.mlp_0(x)

        # Reshape input for Mamba layers
        x = x.view(x.size(0), self.seq_size, self.hidden_size)

        # Process through Mamba layers
        x = self.mamba(x)

        # Reshape output back to original dimensions
        x = x.view(x.size(0), self.latent_size)

        # Pass through final MLP
        x = self.mlp_1(x)

        return x

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
