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
            
def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def kl_divergence_with_free_bits(q_probs, p_probs, free_bits=1.0):
    """
    Compute KL divergence between two categorical distributions with free bits.
    
    Args:
    q_probs: Probabilities of distribution q (B, ...)
    p_probs: Probabilities of distribution p (B, ...)
    free_bits: Minimum KL divergence (default: 1.0)
    
    Returns:
    KL(q||p) for each batch element, clipped at free_bits (B,)
    """
    batch_dim = q_probs.size(0)

    # Add a small epsilon to avoid log(0)
    epsilon = 1e-8
    
    # Compute KL divergence
    kl = q_probs * (torch.log(q_probs + epsilon) - torch.log(p_probs + epsilon))
    
    # Apply free bits
    #kl = torch.max(kl, torch.ones_like(kl) * free_bits)
    
    return kl.mean()
