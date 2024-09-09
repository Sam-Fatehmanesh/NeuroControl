# import torch
# from torch import nn, optim

# #from nfnets.utils import unitwise_norm
# from collections.abc import Iterable

# def unitwise_norm(x: torch.Tensor):
#     if x.ndim <= 1:
#         dim = 0
#         keepdim = False
#     elif x.ndim in [2, 3]:
#         dim = 0
#         keepdim = True
#     elif x.ndim == 4:
#         dim = [1, 2, 3]
#         keepdim = True
#     else:
#         raise ValueError('Wrong input dimensions')

#     return torch.sum(x**2, dim=dim, keepdim=keepdim) ** 0.5


# class AGC(optim.Optimizer):
#     """Generic implementation of the Adaptive Gradient Clipping

#     Args:
#       params (iterable): iterable of parameters to optimize or dicts defining
#             parameter groups
#       optim (torch.optim.Optimizer): Optimizer with base class optim.Optimizer
#       clipping (float, optional): clipping value (default: 1e-3)
#       eps (float, optional): eps (default: 1e-3)
#       model (torch.nn.Module, optional): The original model
#       ignore_agc (str, Iterable, optional): Layers for AGC to ignore
#     """

#     def __init__(self, params, optim: optim.Optimizer, clipping: float = 0.3, eps: float = 1e-3, model=None, ignore_agc=["fc"]):
#         if clipping < 0.0:
#             raise ValueError("Invalid clipping value: {}".format(clipping))
#         if eps < 0.0:
#             raise ValueError("Invalid eps value: {}".format(eps))

#         self.optim = optim

#         defaults = dict(clipping=clipping, eps=eps)
#         defaults = {**defaults, **optim.defaults}

#         if not isinstance(ignore_agc, Iterable):
#             ignore_agc = [ignore_agc]

#         if model is not None:
#             assert ignore_agc not in [
#                 None, []], "You must specify ignore_agc for AGC to ignore fc-like(or other) layers"
#             names = [name for name, module in model.named_modules()]

#             for module_name in ignore_agc:
#                 if module_name not in names:
#                     raise ModuleNotFoundError(
#                         "Module name {} not found in the model".format(module_name))
#             params = [{"params": list(module.parameters())} for name,
#                           module in model.named_modules() if name not in ignore_agc]
        
#         else:
#             params = [{"params": params}]

#         self.agc_params = params
#         self.eps = eps
#         self.clipping = clipping
        
#         self.param_groups = optim.param_groups
#         self.state = optim.state

#         #super(AGC, self).__init__([], defaults)

#     @torch.no_grad()
#     def step(self, closure=None):
#         """Performs a single optimization step.

#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.agc_params:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue

#                 param_norm = torch.max(unitwise_norm(
#                     p.detach()), torch.tensor(self.eps).to(p.device))
#                 grad_norm = unitwise_norm(p.grad.detach())
#                 max_norm = param_norm * self.clipping

#                 trigger = grad_norm > max_norm

#                 clipped_grad = p.grad * \
#                     (max_norm / torch.max(grad_norm,
#                                           torch.tensor(1e-6).to(grad_norm.device)))
#                 p.grad.detach().data.copy_(torch.where(trigger, clipped_grad, p.grad))

#         return self.optim.step(closure)

#     def zero_grad(self, set_to_none: bool = False):
#         r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

#         Arguments:
#             set_to_none (bool): instead of setting to zero, set the grads to None.
#                 This is will in general have lower memory footprint, and can modestly improve performance.
#                 However, it changes certain behaviors. For example:
#                 1. When the user tries to access a gradient and perform manual ops on it,
#                 a None attribute or a Tensor full of 0s will behave differently.
#                 2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
#                 are guaranteed to be None for params that did not receive a gradient.
#                 3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
#                 (in one case it does the step with a gradient of 0 and in the other it skips
#                 the step altogether).
#         """
#         for group in self.agc_params:
#             for p in group['params']:
#                 if p.grad is not None:
#                     if set_to_none:
#                         p.grad = None
#                     else:
#                         if p.grad.grad_fn is not None:
#                             p.grad.detach_()
#                         else:
#                             p.grad.requires_grad_(False)
#                         p.grad.zero_()


import torch
from torch import nn, optim
from typing import Iterable, Union

# def unitwise_norm(x: torch.Tensor):
#     if x.ndim <= 1:
#         return x.abs().max().reshape(1)
#     elif x.ndim == 2:
#         return x.abs().max(dim=1, keepdim=True)[0]
#     elif x.ndim in [3, 4]:
#         return x.abs().reshape(x.shape[0], -1).max(dim=1, keepdim=True)[0].reshape(x.shape[0], *([1] * (x.ndim - 1)))
#     raise ValueError('Tensor dimension not supported')

def unitwise_norm(x: torch.Tensor):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = (1, 2, 3)
        keepdim = True
    else:
        raise ValueError('Tensor dimension not supported')

    return torch.sum(x**2, dim=dim, keepdim=keepdim) ** 0.5


class AGC(optim.Optimizer):
    def __init__(self, params, base_optimizer: optim.Optimizer, clipping: float = 0.3, eps: float = 1e-3, 
                 model: nn.Module = None, ignore_agc: Union[str, Iterable[str]] = ["fc"]):
        if clipping < 0.0:
            raise ValueError(f"Invalid clipping value: {clipping}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")

        self.base_optimizer = base_optimizer
        self.clipping = clipping
        self.eps = eps

        if not isinstance(ignore_agc, Iterable):
            ignore_agc = [ignore_agc]

        if model is not None:
            assert ignore_agc, "You must specify ignore_agc for AGC to ignore fc-like (or other) layers"
            named_modules = dict(model.named_modules())
            for module_name in ignore_agc:
                if module_name not in named_modules:
                    raise ModuleNotFoundError(f"Module name {module_name} not found in the model")
            
            self.agc_params = [
                {'params': list(module.parameters())}
                for name, module in named_modules.items() if name not in ignore_agc
            ]
        else:
            self.agc_params = [{'params': params}]

        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.agc_params:
            for p in group['params']:
                if p.grad is None or p.grad.numel() == 0:
                    continue

                param_norm = unitwise_norm(p)
                grad_norm = unitwise_norm(p.grad)

                if param_norm.numel() == 0 or grad_norm.numel() == 0:
                    continue

                max_norm = torch.clamp(param_norm, min=self.eps) * self.clipping
                trigger = grad_norm > max_norm

                clipped_grad = p.grad * (max_norm / torch.clamp(grad_norm, min=self.eps))
                p.grad.data.copy_(torch.where(trigger, clipped_grad, p.grad))

        return self.base_optimizer.step(closure)



    def zero_grad(self, set_to_none: bool = False):
        for group in self.agc_params:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_()
                        p.grad.zero_()

    def __getstate__(self):
        return {
            'base_optimizer': self.base_optimizer,
            'clipping': self.clipping,
            'eps': self.eps,
            'agc_params': self.agc_params,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
