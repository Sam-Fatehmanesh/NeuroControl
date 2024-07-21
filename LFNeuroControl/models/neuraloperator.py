import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neuralop.models import FNO
import pdb

class FNOPredictor(nn.Module):
    def __init__(self, in_dim, embed_size, heads, ff_dim, out_dim, layer_num):
        super(Transformer, self).__init__()

        self.enLinear = nn.Linear(int(in_dim), embed_size)


        self.FNOs = nn.Sequential()
        for i in range(layer_num):
            self.FNOs.add_module('FNO_' + str(i), FNO(1, 1, modes=16, width=32))

        self.deLinear = nn.Linear(embed_size, int(out_dim))

    def forward(self, x):
        #pdb.set_trace()
        x = self.enLinear(x)
        x = self.FNOs(x)
        x = self.deLinear(x)
        return x


