import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get('t', 1)
                m = state.get('m', torch.zeros_like(p.data, requires_grad=False))
                v = state.get('v', torch.zeros_like(p.data, requires_grad=False))
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1- beta2) * grad ** 2
                a_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # Update parameters
                p.data -= a_t * m / (torch.sqrt(v) + eps)
                # Apply weight decay
                p.data -= lr * weight_decay * p.data

                state['t'] = t + 1
                state['m'] = m
                state['v'] = v

        return loss