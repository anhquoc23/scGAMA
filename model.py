"""Ráp generator + discriminator thành một model hoàn chỉnh."""
from __future__ import annotations

import torch.nn as nn

from models.configs import ModelConfig
from models.Generator import Generator
from models.Discriminator import Discriminator


class Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.generator = Generator(cfg)
        self.discriminator = Discriminator(cfg)