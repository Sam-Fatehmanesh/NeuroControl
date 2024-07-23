from soft_moe_pytorch import SoftMoE, DynamicSlotsSoftMoE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NeuroControl.models.mlp import MLP
import pdb


class MoELayer(nn.Module):
    def __init__(self, dim, ff_dim, num_experts):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts


        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)


        self.mlp = MLP(1, dim, ff_dim, dim)

        self.moe = DynamicSlotsSoftMoE(dim, num_experts=num_experts, geglu=True)

    def forward(self, x):

        residual = x
        x = self.norm1(x)
        x = self.moe(x)

        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x 

        return x

class MoEPredictor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_experts, num_layers):
        super(MoEPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([MoELayer(hidden_dim, hidden_dim*2, num_experts) for _ in range(num_layers)])
        self.in_lin = nn.Linear(in_dim, hidden_dim)
        self.out_lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        seq_dim, _ = x.shape
        #pdb.set_trace()
        x = self.in_lin(x)
        x = x.view(1, seq_dim, self.hidden_dim)

        for layer in self.layers:
            x = layer(x)
        x = x.view(seq_dim, self.hidden_dim)
        x = self.out_lin(x)
        return x
