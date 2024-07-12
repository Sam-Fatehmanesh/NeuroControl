import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from LFNeuroControl.models.transcnn import TransCNN
from LFNeuroControl.models.transformer import STBTransformer
from LFNeuroControl.models.encoder import SpikePositionEncoding


class NeuronControlActor(nn.Module):
    def __init__(self, num_input_frames, image_n, dim, num_trans_layers, num_stim_neurons, stim_time_steps, cnn_kernel_size=3):

        out_dim = stim_time_steps * num_stim_neurons
        self.model = TransCNN(num_input_frames, image_n, dim, out_dim, num_trans_layers)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.act(x)
        return x