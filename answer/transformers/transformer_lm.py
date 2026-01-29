import torch
import torch.nn as nn

from answer.transformers.embedding import Embedding
from answer.transformers.transformer_block import TransformerBlock
from answer.transformers.rms import RMSNorm
from answer.transformers.linear import Linear


class TransformerLM(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        vocab_size: int, 
        context_length: int, 
        num_layers: int,
        theta: float = 10000,
        eps: float = 1e-5,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, theta, eps, device=device, dtype=dtype) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.norm(x)
        x = self.linear(x)

        return x