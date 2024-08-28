import torch
from torch import nn

class STsampleMultiNom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Store input for backward pass
        ctx.save_for_backward(input)
        
        # Sample from multinomial distribution
        x = torch.multinomial(input, 1)
        
        # Convert to one-hot encoded vector
        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, x, 1)

        return one_hot

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output, None

# Straight-Through Multinomial Sampler
class STMNsampler(nn.Module):
    def __init__(self):
        super(STMNsampler, self).__init__()

    def forward(self, x):
            return STsampleMultiNom.apply(x)
            