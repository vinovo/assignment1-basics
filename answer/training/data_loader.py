import numpy as np
import torch

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)
    # starts[:, None] is (batch_size, 1), np.arange(context_length) is (context_length,), 
    indices = starts[:, None] + np.arange(context_length)
    x_batch = torch.tensor(x[indices], device=device)
    y_batch = torch.tensor(x[indices + 1], device=device)
    
    return x_batch, y_batch