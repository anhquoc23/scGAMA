"""Generator = phần decoder + điều phối VAE.

Bản thân Generator CHÍNH LÀ decoder: nó chứa thẳng logic giải nén (from_z ->
cross-attention với gene-identity làm query -> đầu Gamma–Normal). Phần nén nằm
ở Encoder (encoder.py), được Generator gọi như một submodule.

GeneEmbedding tạo một lần trong Generator và chia sẻ cho Encoder (tied), nên
gene-identity lúc nhúng (encoder) và lúc làm query (decoder) là cùng một bảng.

Luồng:  x --Encoder--> (mu, logvar) --reparam--> z
        --decode (chính Generator)--> tham số Gamma–Normal.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .configs import ModelConfig
from .Embedding import GeneEmbedding
from .Attention import MAB, ISAB, PMA
from .GammaNormal import GammaNormalHead
from .Encoder import Encoder


class Generator(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, h = cfg.dim, cfg.num_heads

        # tied gene-identity: dùng chung với Encoder, và làm query khi giải nén
        self.embed = GeneEmbedding(cfg.num_genes, d)

        # phần nén nằm ở Encoder (submodule)
        self.encoder = Encoder(cfg, self.embed)

        # --- phần decoder: chính là Generator ---
        self.from_z = nn.Linear(cfg.latent_dim, d)
        self.decoder = MAB(d, h)             # query = gene-identity, key/value = latent
        self.head = GammaNormalHead(d)       # đầu xác định dropout + tham số phân phối

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, batch_size: int) -> dict:
        zt = self.from_z(z)                              # [B, L, d]
        q = self.embed.gene_query(batch_size)            # [B, G, d]
        h = self.decoder(q, zt)                          # [B, G, d]
        return self.head(h)                              # tham số Gamma–Normal

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)                     # nén
        z = self.reparameterize(mu, logvar)
        params = self.decode(z, x.size(0))               # giải nén (chính Generator)
        return params, mu, logvar