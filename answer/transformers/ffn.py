import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_ff = d_ff = self._dff_from_dmodel(d_model) if not d_ff else d_ff
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

    @staticmethod
    def _dff_from_dmodel(d_model: int) -> int:
        d_ff = d_model * 8 // 3
        remainder = d_ff % 64

        # Find nearest multiple of 64
        # We don't check > 0 because we assume d_model is much larger than 64
        if (remainder < 32):
            d_ff -= remainder
        else:
            d_ff += (64 - remainder)
        
        return d_ff

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self._silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T