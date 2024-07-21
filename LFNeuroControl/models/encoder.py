import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AbsolutePositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(AbsolutePositionEncoding, self).__init__()
        self.d_model = d_model
        
        # Create a long enough P matrix
        pe = torch.zeros(int(max_len), int(d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # Handle the case when d_model is odd
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class SpikePositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SpikePositionEncoding, self).__init__()
        self.position_encoding = AbsolutePositionEncoding(d_model, max_len)
    
    def forward(self, x):
        batch_size, num_neurons, time_steps = x.size()
        
        # Reshape to (time_steps, batch_size * num_neurons)
        x = x.permute(2, 0, 1).reshape(time_steps, batch_size * num_neurons)
        
        # Apply absolute position encoding
        x = self.position_encoding(x)
        
        # Reshape back to (batch_size, num_neurons, time_steps)
        x = x.reshape(time_steps, batch_size, num_neurons).permute(1, 2, 0)
        
        return x