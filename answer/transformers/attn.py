import torch
import torch.nn as nn

from answer.transformers.softmax import softmax

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    weights = softmax(scores, dim=-1)
    return weights @ V


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 8192, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_k = self.d_v = d_model // num_heads
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype

        # include the weights of q, k, v into a single weight tensor
        self.proj_weights = nn.Parameter(torch.empty(3, self.num_heads * self.d_k, self.d_model, device=device, dtype=dtype))
        self.W_O = nn.Parameter(torch.empty(self.d_model, self.num_heads * self.d_v, device=device, dtype=dtype))

        mask = ~torch.triu(torch.ones(max_seq_len, max_seq_len, device=device, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape of x is (batch_size, seq_len, d_model)

        # do projection
        # we need to duplicate x into (batch_size, seq_len, 3, d_model) to project q,k,v at the same time
        x = x.unsqueeze(-2).expand(-1, -1, 3, -1)

        # prepare for projection
        # x has shape (batch_size, seq_len, 3, 1, d_model)
        x = x.unsqueeze(-2)

        # project q,k,v, now shape is (batch_size, seq_len, 3, num_heads * d_k)
        x = x @ self.proj_weights.transpose(-2, -1).squeeze(-2)

        # reshape to (batch_size, seq_len, 3, num_heads, d_k)
        x = x.reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.d_k)

        # permute to (batch_size, num_heads, 3, seq_len, d_k)
        x = x.permute(0, 3, 2, 1, 4)

        # unbind to (batch_size, num_heads, seq_len, d_k)
        q, k, v = x.unbind(dim=-3)
        mask = self.causal_mask[:q.shape[-2], :k.shape[-2]]
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len_q, seq_len_k)
        mask = mask.expand(q.shape[0], q.shape[1], -1, -1)
        # calculate attention, out shape is (batch_size, num_heads, seq_len, d_v)
        attention = scaled_dot_product_attention(q, k, v, mask)

        # reshape into (batch_size, seq_len, num_heads * d_v)
        attention = attention.transpose(1, 2)
        attention = attention.reshape(attention.shape[0], attention.shape[1], self.num_heads * self.d_v)

        # project back, out shape is (batch_size, seq_len, d_model)
        return attention @ self.W_O.T
