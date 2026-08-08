from __future__ import annotations

import torch
import torch.nn as nn


class GeneEmbedding(nn.Module):
    def __init__(self, num_genes: int, dim: int):
        super().__init__()
        self.value_proj = nn.Linear(1, dim)
        self.gene_id = nn.Parameter(torch.randn(num_genes, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, G] -> [B, G, d]
        v = self.value_proj(x.unsqueeze(-1))
        return v + self.gene_id.unsqueeze(0)

    def gene_query(self, batch_size: int) -> torch.Tensor:
        """Query cho decoder: chỉ dùng identity (giá trị đang cần bù khuyết)."""
        return self.gene_id.unsqueeze(0).expand(batch_size, -1, -1)