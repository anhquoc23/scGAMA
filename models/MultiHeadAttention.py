from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim phải chia hết cho num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, d] -> [B, h, N, d/h]
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        B, Nq, D = query.shape
        q = self._split_heads(self.q_proj(query))     # [B, h, Nq, d/h]
        k = self._split_heads(self.k_proj(key))       # [B, h, Nk, d/h]
        v = self._split_heads(self.v_proj(value))     # [B, h, Nk, d/h]

        # scaled dot-product attention (kernel hiệu quả, tự lo scale + softmax)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )                                             # [B, h, Nq, d/h]

        out = out.transpose(1, 2).reshape(B, Nq, D)   # ghép các đầu lại
        return self.out_proj(out)