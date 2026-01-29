import torch
import torch.nn as nn

from answer.transformers.attn import MultiheadAttention
from answer.transformers.ffn import SwiGLU
from answer.transformers.rms import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, 
        num_heads: int, 
        d_ff: int, 
        max_seq_len: int = 8192,
        theta: float = 10000,
        eps: float = 1e-5,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.mha = MultiheadAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.rms1 = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.rms2 = RMSNorm(d_model, eps, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x is (batch_size, seq_len, d_model)
        # create token_positions if not provided
        if token_positions is None:
            token_positions = torch.arange(x.shape[1], device=x.device)
            token_positions = token_positions.unsqueeze(0).expand(x.shape[0], -1)

        # MHA block
        rms_x = self.rms1(x)
        mha_x = self.mha(rms_x, token_positions)
        x = x + mha_x

        # FFN block
        rms_x = self.rms2(x)
        ffn_x = self.ffn(rms_x)
        x = x + ffn_x

        return x