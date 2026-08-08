"""Điểm vào khởi chạy scGAMA — đọc CSV, huấn luyện, xuất CSV đã bù khuyết.

Ví dụ:
    python main.py --input data/expr.csv --output data/expr_imputed.csv --epochs 200

Định dạng CSV đầu vào: hàng = tế bào, cột = gene, có header tên gene và
(tùy chọn) cột đầu là tên/ID tế bào. Đầu ra giữ nguyên tên tế bào + tên gene.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# from scgama import Config, SCGAMA, Trainer
from models.configs import Config
from trainer.train import Trainer
from model import Model
from models.GammaNormal import gamma_normal_expectation, dropout_probability


def read_csv(path: str, index_col) -> pd.DataFrame:
    """Đọc CSV [tế bào × gene]. index_col=0 nếu cột đầu là tên tế bào, None nếu không."""
    df = pd.read_csv(path, index_col=index_col)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df.astype(np.float32)


def build_loaders(x: np.ndarray, cfg: Config):
    """Tách train/val (90/10) và dựng DataLoader để huấn luyện."""
    dataset = TensorDataset(torch.from_numpy(x))
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(cfg.train.seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)
    train_loader = DataLoader(
        train_set, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, shuffle=False)
    return train_loader, val_loader


@torch.no_grad()
def impute_matrix(model: Model, x: np.ndarray, batch_size: int, device) -> np.ndarray:
    """Chạy generator trên toàn bộ ma trận, trả về x̂ (kỳ vọng hỗn hợp Gamma–Normal)."""
    model.eval()
    out = np.empty_like(x)
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).to(device)
        params, _, _ = model.generator(xb)
        out[i : i + batch_size] = gamma_normal_expectation(params).cpu().numpy()
    return out


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Huấn luyện scGAMA + bù khuyết CSV")
    p.add_argument("--input", type=str, required=False, default="./sim.Tung/sim.Tung.drop20/SplatDrop_norm.csv")
    p.add_argument("--output", type=str, default="imputed.csv", help="CSV kết quả")
    p.add_argument("--index_col", type=int, default=0,
                   help="cột tên tế bào (0 = có, dùng -1 nếu CSV không có cột tên)")
    p.add_argument("--log1p", action="store_true",
                   help="áp log1p khi đọc (nếu CSV là count thô)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--latent_dim", type=int, default=None)
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--no_resume", action="store_true", help="bỏ qua resume checkpoint")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = build_args()
    cfg = Config().override(
        epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size, dim=args.dim, latent_dim=args.latent_dim,
        ckpt_dir=args.ckpt_dir, device=args.device,
    )
    if args.no_resume:
        cfg.train.resume = False

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    # ----- đọc CSV -----
    index_col = None if args.index_col < 0 else args.index_col
    df = read_csv(args.input, index_col)
    x = df.values.astype(np.float32)
    if args.log1p:
        x = np.log1p(x)
    cfg.model.num_genes = x.shape[1]          # đồng bộ số gene theo dữ liệu
    print(f"Đọc {args.input}: {x.shape[0]} tế bào × {x.shape[1]} gene")

    # ----- huấn luyện -----
    train_loader, val_loader = build_loaders(x, cfg)
    model = Model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Tham số: {n_params/1e6:.2f}M | device: {cfg.train.device}")
    trainer = Trainer(model, cfg)
    trainer.fit(train_loader, val_loader=val_loader)

    # ----- bù khuyết toàn bộ + xuất CSV -----
    device = next(model.parameters()).device
    x_hat = impute_matrix(model, x, cfg.train.batch_size, device)
    out_df = pd.DataFrame(x_hat, index=df.index, columns=df.columns)
    out_df.to_csv(args.output)
    print(f"Đã ghi ma trận bù khuyết -> {args.output}  ({x_hat.shape[0]} × {x_hat.shape[1]})")

    # (tùy chọn) xuất luôn ma trận xác suất dropout pi
    with torch.no_grad():
        xb = torch.from_numpy(x[: cfg.train.batch_size]).to(device)
        pi = dropout_probability(model.generator(xb)[0])
    print(f"pi dropout trung bình (batch mẫu): {pi.mean():.3f}")


if __name__ == "__main__":
    main()