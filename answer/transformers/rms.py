import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # upcast to prevent overflow when squaring the input
        in_dtype = x.dtype
        x = x.to(torch.float32)

        return (x / self._rms(x) * self.g).to(in_dtype)
