import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from LFNeuroControl.models.transcnn import TransCNN
from LFNeuroControl.models.transformer import STBTransformer
from LFNeuroControl.models.encoder import SpikePositionEncoding


class NeuralControlCritic(nn.Module):
    def __init__(self, num_stim_neurons, stim_time_steps, image_n, num_frames, state_model_dim, action_model_dim):
        super(NeuralControlCritic, self).__init__()

        self.statemodel = TransCNN(num_frames, image_n, state_model_dim, state_model_dim, 1)
        
        self.posEncode = SpikePositionEncoding(num_stim_neurons, max_len=stim_time_steps)
        self.actionmodel = STBTransformer(num_stim_neurons*stim_time_steps, action_model_dim, 32, action_model_dim * 2, action_model_dim, 4)

        control_dim = state_model_dim+action_model_dim

        self.jointmodel = STBTransformer(control_dim, control_dim, 32, control_dim*2, 1, 8)
    
    def forward(self, action, state):
        x_action = self.posEncode(action)
        x_action = self.actionmodel(x_action)

        x_state = self.statemodel(state)

        x = torch.cat((x_state, x_action))

        x = self.jointmodel(x)

        return x
              

