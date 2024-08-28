import torch
from torch import nn

class STsampleMultiNom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.multinomial(input, 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

# Straight-Through Multinomial Sampler
class STMNsampler(nn.Module):
    def __init__(self):
        super(STMNsampler, self).__init__()

    def forward(self, x):
            return STsampleMultiNom.apply(x)