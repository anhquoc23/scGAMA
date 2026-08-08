"""Encoder tách riêng: x -> tham số hậu nghiệm VAE (mu, logvar).

Luồng:  nhúng gene -> nén token (5000 -> 512 -> 256 qua PMA)
        -> self-attention lõi (SAB) -> (mu, logvar) theo từng token.

GeneEmbedding được truyền từ Generator vào để chia sẻ (tied): cùng bảng
gene-identity mà Generator dùng làm query khi giải nén.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .configs import ModelConfig
from .Embedding import GeneEmbedding
from .Attention import SAB, PMA

class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig, embed: GeneEmbedding):
        super().__init__()
        d, h = cfg.dim, cfg.num_heads
        self.embed = embed  # dùng chung với Generator (tied)

        # nén số token xuống dần: mỗi PMA hạ về mức kế tiếp
        self.compress = nn.ModuleList(
            [PMA(d, h, num_seeds=L) for L in cfg.compress_tokens]
        )
        # self-attention trên tập token đã nén (rẻ vì chỉ còn ~256 token)
        self.core = nn.ModuleList([SAB(d, h) for _ in range(cfg.num_core_layers)])

        # bottleneck VAE theo từng token
        self.to_mu = nn.Linear(d, cfg.latent_dim)
        self.to_logvar = nn.Linear(d, cfg.latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.embed(x)                    # [B, G, d]
        for blk in self.compress:
            h = blk(h)                       # -> [B, L, d]
        for blk in self.core:
            h = blk(h)                       # [B, L, d]
        return self.to_mu(h), self.to_logvar(h)   # [B, L, latent]