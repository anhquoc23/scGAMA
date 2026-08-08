from __future__ import annotations

import torch
import torch.nn as nn

from .MultiHeadAttention import MultiHeadAttention


class MAB(nn.Module):

    def __init__(self, dim: int, num_heads: int, ff_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ln0 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Linear(dim * ff_mult, dim)
        )

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        h = self.ln0(Q + self.attn(Q, K, K))
        return self.ln1(h + self.ff(h))


class SAB(nn.Module):

    def __init__(self, dim: int, num_heads: int, **kw):
        super().__init__()
        self.mab = MAB(dim, num_heads, **kw)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)


class ISAB(nn.Module):

    def __init__(self, dim: int, num_heads: int, num_inds: int, **kw):
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(1, num_inds, dim))
        nn.init.xavier_uniform_(self.inducing)
        self.mab0 = MAB(dim, num_heads, **kw)
        self.mab1 = MAB(dim, num_heads, **kw)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.size(0)
        H = self.mab0(self.inducing.expand(B, -1, -1), X)   # [B, m, d]
        return self.mab1(X, H)                              # [B, n, d]


class PMA(nn.Module):

    def __init__(self, dim: int, num_heads: int, num_seeds: int, **kw):
        super().__init__()
        self.seeds = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seeds)
        self.mab = MAB(dim, num_heads, **kw)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.size(0)
        return self.mab(self.seeds.expand(B, -1, -1), X)    # [B, num_seeds, d]