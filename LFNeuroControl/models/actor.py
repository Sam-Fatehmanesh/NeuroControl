import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from LFNeuroControl.models.transcnn import TransCNN
from LFNeuroControl.models.transformer import Transformer
from LFNeuroControl.models.encoder import SpikePositionEncoding


class NeuronControlActor(nn.Module):
    def __init__(self, num_input_frames, image_n, dim, num_trans_layers, num_stim_neurons, stim_time_steps, cnn_kernel_size=3):
        super(NeuronControlActor, self).__init__()
        # Multiplied by two because we need both the mean and std for each action
        self.out_dim = stim_time_steps * num_stim_neurons * 2
        self.model = TransCNN(num_input_frames, image_n, dim, self.out_dim, num_trans_layers)
        self.act = nn.Sigmoid()
        self.action_dims = (num_stim_neurons, stim_time_steps)

    def forward(self, s, state=None, info={}):
        batch_size = s.size()[0] 
        x = self.model(x)
        x = self.act(x)
        MUnSTD = tuple(torch.split(x))
        MUnSTD[0] = MUnSTD.reshape(batch_size, 1, self.action_dims[0], self.action_dims[1])
        MUnSTD[1] = MUnSTD.reshape(batch_size, 1, self.action_dims[0], self.action_dims[1])
        return MUnSTD, state