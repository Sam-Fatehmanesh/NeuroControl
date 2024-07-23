import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from soft_moe_pytorch import SoftMoE, DynamicSlotsSoftMoE
import pdb

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_dim):
        self.embed_dim = embed_size
        super(TransformerDecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        #self.attention = DynamicSlotsSoftMoE(dim=embed_size, num_experts = 8, geglu=True)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_size)
        )
        self.src_mask = None # torch.triu(torch.ones(x.size(0), x.size(0)), diagonal=1)

    def forward(self, x, src_key_padding_mask=None):
        # Setting mask to a simple causual mask
        
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, attn_mask=self.src_mask, key_padding_mask=src_key_padding_mask)
        #pdb.set_trace()
        #x = torch.transpose(x, 0, 1)
        #x = self.attention(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + x 

        return x

class Transformer(nn.Module):
    def __init__(self, in_dim, embed_size, heads, ff_dim, out_dim, layer_num):
        super(Transformer, self).__init__()

        self.enLinear = nn.Linear(int(in_dim), embed_size)


        self.transformer = nn.Sequential()
        for i in range(layer_num):
            self.transformer.add_module("TL"+str(i), TransformerDecoderLayer(embed_size, heads, ff_dim))


        self.deLinear = nn.Linear(embed_size, int(out_dim))

    def forward(self, x):
        #pdb.set_trace()
        x = self.enLinear(x)
        
        x = self.transformer(x)
        x = self.deLinear(x)
        return x


