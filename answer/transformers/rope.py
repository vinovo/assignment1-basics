import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        self.register_buffer("rotation_matrix", self._construct_rotation_matrix(), persistent=False)


    def _construct_rotation_matrix(self):
        R = torch.empty(self.max_seq_len, self.d_k, device=self.device)
        for i in range(self.max_seq_len):
            R_i = torch.zeros(self.d_k, device=self.device)

            for k in range((self.d_k + 1) // 2): # take the ceiling of d_k / 2
                theta_ik = i / (self.theta ** ((2 * k - 2) / self.d_k))
                R_ik = torch.stack([
                    [torch.cos(theta_ik), -torch.sin(theta_ik)],
                    [torch.sin(theta_ik), torch.cos(theta_ik)]
                ])
                R_i[2*k:2*k+2, 2*k:2*k+2] = R_ik
            R[i] = R_i

        return R


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x is (batch_size, seq_len, d_k)
        # rotation matrix is (seq_len, d_k, d_k)
        return x @ self.rotation_matrix[token_positions]