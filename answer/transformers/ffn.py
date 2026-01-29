import torch
import torch.nn as nn

from answer.transformers.linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_ff = d_ff = self._dff_from_dmodel(d_model) if not d_ff else d_ff
        # correspond to W1, W2, W3
        self.ln1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.ln3 = Linear(d_model, d_ff, device=device, dtype=dtype)

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
        return self.ln2(self._silu(self.ln1(x)) * (self.ln3(x)))
