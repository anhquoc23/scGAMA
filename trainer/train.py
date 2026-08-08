"""Vòng huấn luyện Model (Trainer có checkpoint + resume).

Hai giai đoạn:
  1. Warm-up (epoch < warmup_epochs): chỉ NLL tái tạo + KL, chưa bật đối kháng
     -> để generator (yếu vì bottleneck nén + KL) học ra hồn trước.
  2. Đối kháng đầy đủ: mỗi bước cập nhật D rồi cập nhật G, thêm hinge +
     feature matching. R1 áp theo kiểu lazy (mỗi r1_every bước).

AMP: autocast + GradScaler cho phần nặng; NLL Gamma–Normal và R1 tự ép float32
bên trong nên an toàn với fp16.

Checkpoint: lưu 'last.pt' (gồm cả optimizer + scaler + epoch/step) sau mỗi epoch
để resume — nạp lại và bỏ qua các epoch đã train.
"""
from __future__ import annotations

import os

import torch

from models.configs import Config

# from .configs import Config
from model import Model
from models.GammaNormal import gamma_normal_expectation
from trainer import loss as losses


def kl_weight(epoch: int, cfg: Config) -> float:
    """Lịch KL annealing tuyến tính: 0 -> lambda_kl trong kl_anneal_epochs epoch."""
    if cfg.train.kl_anneal_epochs <= 0:
        return cfg.train.lambda_kl
    frac = min(1.0, epoch / cfg.train.kl_anneal_epochs)
    return cfg.train.lambda_kl * frac


class Trainer:
    def __init__(self, model: Model, cfg: Config):
        self.cfg = cfg
        self.tcfg = cfg.train
        self.device = torch.device(
            self.tcfg.device if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)

        self.opt_g = torch.optim.Adam(
            model.generator.parameters(), lr=self.tcfg.lr_g, betas=self.tcfg.betas
        )
        self.opt_d = torch.optim.Adam(
            model.discriminator.parameters(), lr=self.tcfg.lr_d, betas=self.tcfg.betas
        )
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.tcfg.use_amp and self.device.type == "cuda"
        )
        self.start_epoch = 0
        self.step = 0

    # ---------- checkpoint ----------
    def save_checkpoint(self, epoch: int, tag: str = "last") -> None:
        os.makedirs(self.tcfg.ckpt_dir, exist_ok=True)
        path = os.path.join(self.tcfg.ckpt_dir, f"{tag}.pt")
        torch.save(
            {
                "epoch": epoch,
                "step": self.step,
                "model": self.model.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt_d": self.opt_d.state_dict(),
                "scaler": self.scaler.state_dict(),
                "config": self.cfg.to_dict(),
            },
            path,
        )

    def maybe_resume(self) -> None:
        path = os.path.join(self.tcfg.ckpt_dir, "last.pt")
        if not (self.tcfg.resume and os.path.isfile(path)):
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]
        self.start_epoch = ckpt["epoch"] + 1
        print(f"Resume từ '{path}': tiếp tục từ epoch {self.start_epoch}")

    # ---------- một bước cập nhật ----------
    def discriminator_step(self, x) -> dict:
        self.opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            params, _, _ = self.model.generator(x)
            x_fake = gamma_normal_expectation(params)

        with torch.autocast(device_type=x.device.type, enabled=self.tcfg.use_amp):
            loss_d, parts = losses.discriminator_loss(
                self.model.discriminator,
                x,
                x_fake,
                r1_gamma=self.tcfg.r1_gamma,
                r1_every=self.tcfg.r1_every,
                step=self.step,
            )

        self.scaler.scale(loss_d).backward()
        self.scaler.unscale_(self.opt_d)
        torch.nn.utils.clip_grad_norm_(
            self.model.discriminator.parameters(), self.tcfg.grad_clip
        )
        self.scaler.step(self.opt_d)
        self.scaler.update()
        return parts

    def generator_step(self, x, kl_w: float, adversarial: bool) -> dict:
        self.opt_g.zero_grad(set_to_none=True)
        with torch.autocast(device_type=x.device.type, enabled=self.tcfg.use_amp):
            params, mu, logvar = self.model.generator(x)
            x_fake = gamma_normal_expectation(params)
            loss_g, parts = losses.generator_loss(
                x, params, mu, logvar,
                discriminator=self.model.discriminator,
                fake_x=x_fake,
                lambda_rec=self.tcfg.lambda_rec,
                kl_weight=kl_w,
                lambda_fm=self.tcfg.lambda_fm,
                adversarial=adversarial,
            )

        self.scaler.scale(loss_g).backward()
        self.scaler.unscale_(self.opt_g)
        torch.nn.utils.clip_grad_norm_(
            self.model.generator.parameters(), self.tcfg.grad_clip
        )
        self.scaler.step(self.opt_g)
        self.scaler.update()
        return parts

    # ---------- validate ----------
    @torch.no_grad()
    def validate(self, loader) -> dict:
        """RMSE/MAE giữa x và x̂ (kỳ vọng hỗn hợp). Theo dõi mean-regression.

        Đánh giá imputation cuối cùng nên dùng entry bị che held-out; đây là
        chỉ số theo dõi trong lúc train.
        """
        self.model.eval()
        se = ae = n = 0.0
        for batch in loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(self.device)
            params, _, _ = self.model.generator(x)
            x_hat = gamma_normal_expectation(params)
            se += torch.sum((x_hat - x) ** 2).item()
            ae += torch.sum((x_hat - x).abs()).item()
            n += x.numel()
        return {"rmse": (se / n) ** 0.5, "mae": ae / n}

    # ---------- vòng chính ----------
    def fit(self, loader, val_loader=None) -> Model:
        self.maybe_resume()
        for epoch in range(self.start_epoch, self.tcfg.epochs):
            adversarial = epoch >= self.tcfg.warmup_epochs
            kl_w = kl_weight(epoch, self.cfg)
            self.model.train()

            agg = {"rec": 0.0, "kl": 0.0, "d_hinge": 0.0, "n": 0}
            for batch in loader:
                x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(self.device)
                if adversarial:
                    dp = self.discriminator_step(x)
                    agg["d_hinge"] += float(dp["d_hinge"])
                gp = self.generator_step(x, kl_w, adversarial)
                agg["rec"] += float(gp["rec"])
                agg["kl"] += float(gp["kl"])
                agg["n"] += 1
                self.step += 1

            n = max(agg["n"], 1)
            phase = "adv " if adversarial else "warm"
            line = (
                f"[{epoch:3d}] ({phase}) rec={agg['rec']/n:.4f} "
                f"kl={agg['kl']/n:.4f} d={agg['d_hinge']/n:.4f} kl_w={kl_w:.3f}"
            )
            if val_loader is not None:
                m = self.validate(val_loader)
                line += f"  val_rmse={m['rmse']:.4f} val_mae={m['mae']:.4f}"
            print(line)

            self.save_checkpoint(epoch, tag="last")
            if self.tcfg.save_every > 0 and (epoch + 1) % self.tcfg.save_every == 0:
                self.save_checkpoint(epoch, tag=f"epoch{epoch:04d}")
        return self.model


def fit(model: Model, loader, cfg: Config, val_loader=None) -> Model:
    """Tiện ích giữ tương thích với main.py: dựng Trainer rồi chạy."""
    return Trainer(model, cfg).fit(loader, val_loader=val_loader)