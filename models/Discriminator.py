
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from .configs import ModelConfig
from .Embedding import GeneEmbedding
from .Attention import ISAB, PMA


def apply_spectral_norm(module: nn.Module) -> None:
    """Bọc spectral norm lên mọi nn.Linear con (đệ quy) — gồm cả Q/K/V trong MHA."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, spectral_norm(child))
        else:
            apply_spectral_norm(child)


class Discriminator(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, h = cfg.dim, cfg.num_heads
        self.embed = GeneEmbedding(cfg.num_genes, d)

        self.isab = nn.ModuleList(
            [ISAB(d, h, num_inds=cfg.num_inds) for _ in range(cfg.num_isab)]
        )
        self.pma = PMA(d, h, num_seeds=1)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        apply_spectral_norm(self)   # phủ toàn bộ Linear, kể cả bên trong attention

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        h = self.embed(x)                       # [B, G, d]
        for blk in self.isab:
            h = blk(h)                          # [B, G, d]
        feat = self.pma(h).squeeze(1)           # [B, d]
        score = self.head(feat).squeeze(-1)     # [B]
        if return_feature:
            return score, feat
        return score