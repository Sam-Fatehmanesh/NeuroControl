import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NeuroControl.models.mlp import MLP
from mamba_ssm import Mamba2
import pdb
import copy

class NeuralControlCritic(nn.Module):
    def __init__(self, hidden_state_size, mamba_seq_size, reward_size, reward_prediction_logits_num=41):
        super(NeuralControlCritic, self).__init__()

        assert hidden_state_size % mamba_seq_size == 0, "hidden_state_size must be divisible by mamba_seq_size"

        #self.loss = nn.MSELoss()

        self.mamba_seq_size = mamba_seq_size
        self.hidden_size = (hidden_state_size // mamba_seq_size)*8

        self.reward_prediction_logits_num = reward_prediction_logits_num
        self.reward_size = reward_size
        # # Assert that hidden_size is divisible by mamba_seq_size
        # assert hidden_size % mamba_seq_size == 0, "hidden_size must be divisible by mamba_seq_size"

        #self.loss = nn.MSELoss()

        self.mlp_in = MLP(2, hidden_state_size, hidden_state_size*self.mamba_seq_size, self.hidden_size*self.mamba_seq_size)
        self.mamba = nn.Sequential(
            #Mamba2(self.hidden_size),
            Mamba2(self.hidden_size),
        )
        #self.pre_mlp_flat = nn.Flatten(start_dim=0)
        self.mlp_out = MLP(2, self.hidden_size*self.mamba_seq_size, hidden_state_size, reward_size*reward_prediction_logits_num)

        self.ema_decay = 0.98
        self.ema_critic = copy.deepcopy(self)
        


    def forward(self, x, ema_forward=True):
        batch_dim = x.shape[0]

        ema_x = x
        
        # Regular forward pass
        x = self.mlp_in(x)
        x = x.view(batch_dim, self.mamba_seq_size, self.hidden_size)
        x = self.mamba(x)
        x = x.view(batch_dim, self.mamba_seq_size*self.hidden_size)
        output = self.mlp_out(x).view(batch_dim, self.reward_size, self.reward_prediction_logits_num)
        
        # EMA forward pass (with no gradients)
        if ema_forward:
            with torch.no_grad():
                ema_x = self.ema_critic.mlp_in(ema_x)
                ema_x = ema_x.view(batch_dim, self.mamba_seq_size, self.hidden_size)
                ema_x = self.ema_critic.mamba(ema_x)
                ema_x = ema_x.view(batch_dim, self.mamba_seq_size*self.hidden_size)
                ema_output = self.ema_critic.mlp_out(ema_x).view(batch_dim, self.reward_size, self.reward_prediction_logits_num)
            
            return output, ema_output.detach()
        else:
            return output


    # def loss(self, a, b):
    #     return torch.sum(torch.square(a-b))


    def update_ema(self):
        for param, ema_param in zip(self.parameters(), self.ema_critic.parameters()):
            ema_param.data = self.ema_decay * ema_param.data + (1 - self.ema_decay) * param.data