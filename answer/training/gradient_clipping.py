import torch
from collections.abc import Iterable

def clip_gradient(params: Iterable[torch.nn.Parameter], M: float, eps: float = 1e-6):
    grad_sum = torch.tensor(0.0)
    for param in params:
        if param.grad is None:
            continue

        grad_sum += torch.sum(torch.square(param.grad))
    
    l2_norm = torch.sqrt(grad_sum)
    if l2_norm > M:
        for param in params:
            if param.grad is None:
                continue

            param.grad = param.grad * M / (l2_norm + eps)
