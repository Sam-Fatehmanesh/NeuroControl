import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_dim):
        super(TransformerDecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_size)
        )

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = residual + x  

        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + x 

        return x

class STBTransformer(nn.Module):
    def __init__(self, in_dim, embed_size, heads, ff_dim, out_dim, layer_num):
        super(STBTransformer, self).__init__()

        self.transformer = nn.Sequential()
        for _ in range(layer_num):
            self.transformer.add_module(TransformerDecoderLayer(embed_size, heads, ff_dim))

        self.enLinear = nn.Linear(in_dim, embed_size)
        self.deLinear = nn.Linear(embed_size, out_dim)

    def forward(self, x):
        x = self.enLinear(x)
        x = self.transformer(x)
        x = self.deLinear(x)
        return x


