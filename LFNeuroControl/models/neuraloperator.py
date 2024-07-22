import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neuralop.models import TFNO, FNO
import pdb
import math

def has_integer_square_root(n):
    sqrt_n = int(math.sqrt(n))
    return sqrt_n * sqrt_n == n

class FNOPredictor(nn.Module):
    def __init__(self, in_dim, embed_size, out_dim, no_layers=4):
        super(FNOPredictor, self).__init__()

        assert has_integer_square_root(embed_size), f"{embed_size} must have an integer square root."


        self.embed_size = embed_size
        self.sqrt_embed_size = int(math.sqrt(embed_size))
        self.enLinear = nn.Linear(int(in_dim), embed_size)

        self.fno = FNO(n_modes=(1,), hidden_channels=1, in_channels=1, lifting_channels=embed_size, projection_channels=embed_size, n_layers=no_layers)

        # self.FNOs = nn.Sequential()
        # for i in range(layer_num):
        #     self.FNOs.add_module('FNO_' + str(i), FNO(1, 1, modes=16, width=32))

        self.deLinear = nn.Linear(embed_size, int(out_dim))

    def forward(self, x):
        #pdb.set_trace()
        x = self.enLinear(x)

        x = x.view(1, 1, self.embed_size)
        #print("####")
        x = self.fno(x)
        x = x.view(1,1, 1, self.embed_size)

        x = self.deLinear(x)
        return x

# class FNOMLPPredictor(nn.Module):
#     def __init__(self, in_dim, embed_size, heads, ff_dim, out_dim, layer_num):
#         super(Transformer, self).__init__()

#         self.enLinear = nn.Linear(int(in_dim), embed_size)

#         self.fno = TFNO1d(n_modes_height=16, hidden_channels=64)

#         # self.FNOs = nn.Sequential()
#         # for i in range(layer_num):
#         #     self.FNOs.add_module('FNO_' + str(i), FNO(1, 1, modes=16, width=32))

#         self.deLinear = nn.Linear(embed_size, int(out_dim))

#     def forward(self, x):
#         #pdb.set_trace()

#         x = self.enLinear(x)
#         x = self.fno(x)
#         x = self.deLinear(x)

#         return x