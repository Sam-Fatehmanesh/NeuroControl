import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NeuroControl.models.mlp import MLP
from mamba_ssm import Mamba2
import pdb
import copy

class NeuralControlCritic(nn.Module):
    def __init__(self, model_state_size, mamba_seq_size, reward_size, internal_h_state_multiplier = 1, reward_prediction_logits_num=41):
        super(NeuralControlCritic, self).__init__()

        assert model_state_size % mamba_seq_size == 0, "model_state_size must be divisible by mamba_seq_size"

        #self.loss = nn.MSELoss()

        self.mamba_seq_size = mamba_seq_size
        self.mamba_embed_size = ((((2 ** ((int(model_state_size) - 1).bit_length() - 1)) // mamba_seq_size)))*internal_h_state_multiplier

        self.reward_prediction_logits_num = reward_prediction_logits_num
        self.reward_size = reward_size
        # # Assert that hidden_size is divisible by mamba_seq_size
        # assert hidden_size % mamba_seq_size == 0, "hidden_size must be divisible by mamba_seq_size"

        #self.loss = nn.MSELoss()

        self.mlp_in = MLP(2, model_state_size, self.mamba_embed_size*self.mamba_seq_size, self.mamba_embed_size*self.mamba_seq_size)
        self.mamba = nn.Sequential(
            Mamba2(self.mamba_embed_size),
            Mamba2(self.mamba_embed_size),
        )

        self.mlp_out = MLP(1, self.mamba_embed_size*self.mamba_seq_size, self.mamba_embed_size*self.mamba_seq_size, reward_size*reward_prediction_logits_num)

        self.ema_decay = 0.98
        self.ema_critic = copy.deepcopy(self)
        


    def forward(self, x, ema_forward=True):
        batch_dim = x.shape[0]

        ema_x = x
        
        # Regular forward pass
        x = self.mlp_in(x)
        x = x.view(batch_dim, self.mamba_seq_size, self.mamba_embed_size)
        x = self.mamba(x)
        x = x.view(batch_dim, self.mamba_seq_size*self.mamba_embed_size)
        output = self.mlp_out(x).view(batch_dim, self.reward_size, self.reward_prediction_logits_num)
        
        # EMA forward pass (with no gradients)
        if ema_forward:
            with torch.no_grad():
                ema_x = self.ema_critic.mlp_in(ema_x)
                ema_x = ema_x.view(batch_dim, self.mamba_seq_size, self.mamba_embed_size)
                ema_x = self.ema_critic.mamba(ema_x)
                ema_x = ema_x.view(batch_dim, self.mamba_seq_size*self.mamba_embed_size)
                ema_output = self.ema_critic.mlp_out(ema_x).view(batch_dim, self.reward_size, self.reward_prediction_logits_num)

            return output, ema_output.detach()
        else:
            return output


    def update_ema(self):
        for param, ema_param in zip(self.parameters(), self.ema_critic.parameters()):
            ema_param.data = self.ema_decay * ema_param.data + (1 - self.ema_decay) * param.data

    