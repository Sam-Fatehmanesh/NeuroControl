import torch
from torch import nn
import pdb
import torch.nn.functional as F
from torch.distributions import Distribution, Independent, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence

class STsampleMultiNom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Store input for backward pass
        ctx.save_for_backward(input)
        
        # Sample from multinomial distribution
        #pdb.set_trace()
        x = torch.multinomial(input, 1)
        
        # Convert to one-hot encoded vector
        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, x, 1)

        # replace sampling and one_hot scatter with OneHotCategoricalStraightThrough

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

def kl_divergence_with_free_bits(q_probs, p_probs, batch_size, free_bits=1.0):
    """
    Compute KL divergence between two categorical distributions with free bits.
    
    Args:
    q_probs: Probabilities of distribution q (B, ...)
    p_probs: Probabilities of distribution p (B, ...)
    free_bits: Minimum KL divergence (default: 1.0)
    
    Returns:
    KL(q||p) for each batch element, clipped at free_bits (B,)
    """
    # pdb.set_trace()
    # kl = kl_divergence(
    #     Independent(OneHotCategoricalStraightThrough(logits=q_probs), 1),
    #     Independent(OneHotCategoricalStraightThrough(logits=p_probs), 1),
    # )
    # free_nats = torch.full_like(kl, free_bits)
    # kl =  torch.maximum(kl, free_nats)

    # return kl.mean()


    batch_dim = q_probs.size(0)

    # Add a small epsilon to avoid log(0)
    epsilon = 1e-15
    
    # Compute KL divergence
    kld = q_probs * (torch.log(q_probs + epsilon) - torch.log(p_probs + epsilon))
    
    # Apply free bits
    #fbs = torch.full_like(kld, free_bits)
    #kld = torch.max(kld, fbs)


    
    #kl = torch.max(kl, torch.ones_like(kl) * free_bits)
    #kld = kld.sum() / batch_dim

    # perform the .max for each element
    #kld = torch.max(kld, torch.tensor(free_bits).to(kld.device))
    #kld=kld.mean()
    # kld = kld.view(batch_size, -1).sum(dim=1)
    #kld = torch.max(kld, torch.tensor(free_bits).to(kld.device) / batch_size)
    return kld.mean()



class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

def symlogMSE(x, y):
    return F.mse_loss(symlog(x), symlog(y))




def twohot_symexp_loss(predicted_logits, true_values, num_bins=41):
    # Ensure inputs are 2D
    if predicted_logits.dim() == 1:
        predicted_logits = predicted_logits.unsqueeze(0)
    if true_values.dim() == 0:
        true_values = true_values.unsqueeze(0)
    elif true_values.dim() == 1:
        true_values = true_values.unsqueeze(1)

    batch_size = true_values.shape[0]

    # Create exponentially spaced bins
    bins = symexp(torch.linspace(-20, 20, num_bins)).to(true_values.device)
    
    # Compute twohot encoding for true values
    true_symlog = symlog(true_values)
    k = torch.sum(bins < true_symlog, dim=1).long()
    k = torch.clamp(k, 0, num_bins - 2)
    
    lower_bin = bins[k]
    upper_bin = bins[k + 1]
    
    # Compute weights for twohot encoding
    weight_upper = (true_symlog.squeeze() - lower_bin) / (upper_bin - lower_bin)
    weight_lower = 1 - weight_upper
    
    # Create twohot encoding
    twohot = torch.zeros_like(predicted_logits)
    twohot.scatter_(1, k.unsqueeze(1), weight_lower.unsqueeze(1))
    twohot.scatter_(1, (k + 1).unsqueeze(1), weight_upper.unsqueeze(1))
    
    # # Add uniform mixture
    # epsilon = 0.01
    # twohot = (1 - epsilon) * twohot + epsilon / num_bins
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(predicted_logits, twohot, reduction='mean')
    
    # Compute predicted values
    softmax_probs = F.softmax(predicted_logits, dim=1)
    
    # Separate positive and negative bins
    pos_mask = bins >= 0
    neg_mask = bins < 0
    
    # Compute expected prediction
    pos_pred = torch.sum(softmax_probs[:, pos_mask] * bins[pos_mask], dim=1)
    neg_pred = torch.sum(softmax_probs[:, neg_mask] * bins[neg_mask], dim=1)
    predicted_values = pos_pred + neg_pred
    
    return loss, predicted_values