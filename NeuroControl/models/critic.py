import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NeuroControl.models.transcnn import TransCNN
from NeuroControl.models.transformer import Transformer
from NeuroControl.models.encoder import SpikePositionEncoding


class NeuralControlCritic(nn.Module):
    def __init__(self, num_stim_neurons, stim_time_steps, image_n, num_frames, state_model_dim, action_model_dim):
        super(NeuralControlCritic, self).__init__()

        self.statemodel = TransCNN(num_frames, image_n, state_model_dim, state_model_dim, 1)
        
        self.posEncode = SpikePositionEncoding(num_stim_neurons, max_len=stim_time_steps)
        self.actionmodel = Transformer(num_stim_neurons*stim_time_steps, action_model_dim, 32, action_model_dim * 2, action_model_dim, 4)

        critic_dim = state_model_dim+action_model_dim

        self.jointmodel = Transformer(critic_dim, critic_dim, 32, critic_dim*2, 1, 8)
    
    def forward(self, s, a):
        x_action = self.posEncode(a)
        x_action = self.actionmodel(a)

        x_state = self.statemodel(s)

        x = torch.cat((x_state, x_action))

        x = self.jointmodel(x)

        return x