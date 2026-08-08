"""Cấu hình cho scGAMA.

Toàn bộ siêu tham số gom vào các dataclass thuần Python (không dùng YAML),
để import trực tiếp và ghi đè bằng tham số dòng lệnh trong main.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    # --- chung ---
    num_genes: int = 2000          # G — số gene (đặt lại theo dữ liệu thực)
    dim: int = 128                 # d — chiều embedding của mỗi token
    num_heads: int = 4             # số đầu của Multi-Head Attention

    # --- generator ---
    compress_tokens: tuple = (512, 256)   # nén token: 5000 -> 512 -> 256
    num_core_layers: int = 2       # số lớp self-attention trên tập token đã nén
    latent_dim: int = 64           # chiều latent của VAE (mỗi token)

    # --- discriminator ---
    num_inds: int = 32             # m — số điểm cảm ứng của ISAB
    num_isab: int = 1              # độ nông của discriminator (1–2 khối)


@dataclass
class TrainConfig:
    epochs: int = 200
    warmup_epochs: int = 20        # giai đoạn 1: chỉ tái tạo + KL (chưa đối kháng)
    batch_size: int = 128

    # TTUR — để D không áp đảo generator yếu ở giai đoạn đầu, lr của D thấp hơn
    lr_g: float = 2e-4
    lr_d: float = 1e-4
    betas: tuple = (0.5, 0.9)

    # trọng số các hàm mất mát
    lambda_rec: float = 1.0        # NLL Gamma–Normal
    lambda_kl: float = 1.0         # KL cực đại (sau annealing)
    lambda_fm: float = 10.0        # feature matching
    kl_anneal_epochs: int = 30     # số epoch để KL tăng dần 0 -> lambda_kl

    # regularization discriminator
    r1_gamma: float = 10.0
    r1_every: int = 16             # lazy R1: chỉ phạt mỗi 16 bước

    use_amp: bool = True
    grad_clip: float = 1.0
    device: str = "cuda"
    seed: int = 42

    ckpt_dir: str = "checkpoints"
    resume: bool = True
    save_every: int = 10


@dataclass
class DataConfig:
    path: str = ""                 # đường dẫn .h5ad hoặc .npz; rỗng = dữ liệu giả để smoke-test
    normalize: bool = True         # chuẩn hóa + log1p nếu đọc count thô
    n_top_genes: int = 2000        # số HVG giữ lại


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def override(self, **kwargs) -> "Config":
        """Ghi đè phẳng: override(epochs=50, dim=256, path='x.h5ad')."""
        for key, val in kwargs.items():
            if val is None:
                continue
            for group in (self.model, self.train, self.data):
                if hasattr(group, key):
                    setattr(group, key, val)
                    break
        return self


def default_config() -> Config:
    return Config()