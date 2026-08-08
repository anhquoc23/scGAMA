"""Các hàm mất mát cho vòng huấn luyện đối kháng.

Thành phần cơ bản:
  - hinge_d / hinge_g : loss đối kháng dạng hinge (ổn định hơn BCE)
  - r1_penalty        : phạt gradient trên mẫu thật (tính float32, dùng lazy)
  - kl_divergence     : KL của VAE so với N(0, I)
  - feature_matching  : khớp đặc trưng discriminator giữa thật và giả

Hàm gộp (dồn toàn bộ phép tính mất mát về đây, train.py chỉ lo tối ưu):
  - discriminator_loss : hinge (+ lazy R1)
  - generator_loss     : NLL Gamma–Normal + KL (+ adversarial + feature matching)

NLL tái tạo lấy từ gamma_normal.gamma_normal_nll.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.GammaNormal import gamma_normal_nll


# ---------- thành phần cơ bản ----------
def hinge_d(real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()


def hinge_g(fake_score: torch.Tensor) -> torch.Tensor:
    return -fake_score.mean()


def r1_penalty(discriminator, real_x: torch.Tensor) -> torch.Tensor:
    """R1: 0.5 * ||grad_x D(x_real)||^2. Tính ngoài autocast, float32."""
    real_x = real_x.detach().float().requires_grad_(True)
    with torch.autocast(device_type=real_x.device.type, enabled=False):
        score = discriminator(real_x)
        grad = torch.autograd.grad(
            outputs=score.sum(), inputs=real_x, create_graph=True
        )[0]
    return 0.5 * grad.pow(2).flatten(1).sum(dim=1).mean()


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # KL[N(mu,sigma^2) || N(0,1)], trung bình theo batch, tổng theo chiều còn lại
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=tuple(range(1, kl.dim()))).mean()


def feature_matching(feat_real: torch.Tensor, feat_fake: torch.Tensor) -> torch.Tensor:
    # khớp đặc trưng thật/giả (feat_real coi như hằng số với generator)
    return F.mse_loss(feat_fake, feat_real.detach())


# ---------- hàm gộp ----------
def discriminator_loss(
    discriminator, real_x, fake_x, r1_gamma=0.0, r1_every=1, step=0
):
    """Loss cập nhật discriminator: hinge (+ lazy R1).

    fake_x nên được sinh sẵn dưới no_grad ở phía trainer.
    """
    real_score = discriminator(real_x)
    fake_score = discriminator(fake_x)
    loss = hinge_d(real_score, fake_score)
    parts = {"d_hinge": loss.detach()}

    if r1_gamma > 0 and step % r1_every == 0:
        # bù hệ số r1_every vì chỉ phạt thưa (lazy regularization)
        reg = r1_penalty(discriminator, real_x)
        loss = loss + r1_gamma * r1_every * reg
        parts["r1"] = reg.detach()
    return loss, parts


def generator_loss(
    x,
    params,
    mu,
    logvar,
    discriminator=None,
    fake_x=None,
    lambda_rec: float = 1.0,
    kl_weight: float = 0.0,
    lambda_fm: float = 10.0,
    adversarial: bool = False,
    mask=None,
):
    """Loss cập nhật generator.

    Warm-up (adversarial=False): chỉ NLL Gamma–Normal + KL.
    Đối kháng (adversarial=True): thêm hinge (đánh lừa D) + feature matching.
    mask: nếu có, NLL chỉ tính trên entry được quan sát.
    """
    loss_rec = gamma_normal_nll(x, params, mask=mask)   # tự ép fp32 bên trong
    loss_kl = kl_divergence(mu, logvar)
    total = lambda_rec * loss_rec + kl_weight * loss_kl
    parts = {"rec": loss_rec.detach(), "kl": loss_kl.detach()}

    if adversarial:
        fake_score, feat_fake = discriminator(fake_x, return_feature=True)
        with torch.no_grad():
            _, feat_real = discriminator(x, return_feature=True)
        loss_adv = hinge_g(fake_score)
        loss_fm = feature_matching(feat_real, feat_fake)
        total = total + loss_adv + lambda_fm * loss_fm
        parts["adv"] = loss_adv.detach()
        parts["fm"] = loss_fm.detach()

    parts["total"] = total.detach()
    return total, parts