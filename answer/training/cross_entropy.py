import torch

# logits: (B, vocab_size)
# targets: (B,)
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    # subtract by the max value for numerical stability
    logits = logits - logits.max(dim=-1, keepdim=True).values

    # -log softmax(x_{1:i})[x_{i+1}]
    # = -log( exp(o_i[x_{i+1}]) / sum_a exp(o_i[a]) )
    # = -o_i[x_{i+1}] + log(sum_a exp(o_i[a]))   # since log(exp(z)) = z

    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1))
    sum_exp = torch.sum(torch.exp(logits), dim=-1)
    loss_per_sample = -target_logits + torch.log(sum_exp)
    
    return loss_per_sample.mean()
