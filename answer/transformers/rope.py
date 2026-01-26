import torch
import torch.nn as nn

class RoPE(nn.Module):
    # Class-level cache: maps (theta, d_k, max_seq_len, device) -> rotation_matrix
    _cache = {}
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # Create cache key from parameters
        cache_key = (theta, d_k, max_seq_len, str(device))
        
        # Check if rotation matrix already exists in cache
        if cache_key in RoPE._cache:
            rotation_matrix = RoPE._cache[cache_key]
        else:
            # Construct and cache it
            rotation_matrix = self._construct_rotation_matrix()
            RoPE._cache[cache_key] = rotation_matrix
        
        # Register the (possibly shared) buffer
        self.register_buffer("rotation_matrix", rotation_matrix, persistent=False)


    def _construct_rotation_matrix(self):
        positions = torch.arange(self.max_seq_len, device=self.device).unsqueeze(1)
        dims = torch.arange(self.d_k // 2, device=self.device)
        theta_ik = positions / self.theta ** (2 * dims / self.d_k)
        cos_theta_ik = torch.cos(theta_ik)
        sin_theta_ik = torch.sin(theta_ik)

        R = torch.zeros(self.max_seq_len, self.d_k, self.d_k, device=self.device)
        R[:, 2*dims, 2*dims] = cos_theta_ik
        R[:, 2*dims, 2*dims+1] = -sin_theta_ik
        R[:, 2*dims+1, 2*dims] = sin_theta_ik
        R[:, 2*dims+1, 2*dims+1] = cos_theta_ik

        return R

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x is (batch_size, seq_len, d_k)
        # token_positions is (seq_len,)
        # self.rotation_matrix[token_positions] is (seq_len, d_k, d_k)
        # Transpose to get R^T for row vector multiplication: x @ R^T
        R = self.rotation_matrix[token_positions].transpose(-2, -1).unsqueeze(0)  # (1, seq_len, d_k, d_k)
        return (x.unsqueeze(-2) @ R).squeeze(-2)