"""Mô hình hóa dropout bằng hỗn hợp Gamma–Normal (theo scImpute).

Với mỗi (tế bào, gene), decoder xuất tham số của một hỗn hợp hai thành phần:
  - Gamma(alpha, beta): thành phần giá trị gần 0  -> dropout kỹ thuật
  - Normal(mu, sigma) : thành phần biểu hiện thật
  - pi in [0, 1]      : trọng số thành phần Gamma = XÁC SUẤT DROPOUT

Hàm mất mát tái tạo là negative log-likelihood của hỗn hợp (thay cho MSE),
nên model được phép nói 'entry này là dropout' thay vì bị ép tái tạo số 0.

Lưu ý số học: mật độ Gamma phân kỳ khi x -> 0, và alpha/beta/sigma phải > 0.
Vì vậy NLL được tính hoàn toàn trong float32, ngoài autocast, có clamp cận dưới.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_2PI = math.log(2.0 * math.pi)


class GammaNormalHead(nn.Module):
    """Từ đặc trưng decoder [B, G, d] -> tham số hỗn hợp Gamma–Normal theo gene."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.to_pi = nn.Linear(dim, 1)      # gate dropout
        self.to_alpha = nn.Linear(dim, 1)   # Gamma shape
        self.to_beta = nn.Linear(dim, 1)    # Gamma rate
        self.to_mu = nn.Linear(dim, 1)      # Normal mean
        self.to_sigma = nn.Linear(dim, 1)   # Normal std

    def forward(self, h: torch.Tensor) -> dict:
        eps = self.eps
        pi = torch.sigmoid(self.to_pi(h)).squeeze(-1)              # [B, G]
        alpha = F.softplus(self.to_alpha(h)).squeeze(-1) + eps
        beta = F.softplus(self.to_beta(h)).squeeze(-1) + eps
        mu = self.to_mu(h).squeeze(-1)
        sigma = F.softplus(self.to_sigma(h)).squeeze(-1) + eps
        return {"pi": pi, "alpha": alpha, "beta": beta, "mu": mu, "sigma": sigma}


def gamma_normal_nll(
    x: torch.Tensor, params: dict, mask: torch.Tensor | None = None, eps: float = 1e-6
) -> torch.Tensor:
    """NLL trung bình của x dưới hỗn hợp Gamma–Normal.

    mask (nếu có): chỉ tính NLL trên các entry được quan sát (mask==1) — dùng khi
    huấn luyện theo kiểu che dropout để đánh giá imputation.
    """
    # ép float32 và tách khỏi autocast để tránh NaN từ fp16
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float().clamp_min(eps)
        pi = params["pi"].float().clamp(eps, 1.0 - eps)
        alpha = params["alpha"].float()
        beta = params["beta"].float()
        mu = params["mu"].float()
        sigma = params["sigma"].float().clamp_min(eps)

        # log mật độ Gamma (shape-rate), x > 0
        log_gamma = (
            alpha * torch.log(beta)
            - torch.lgamma(alpha)
            + (alpha - 1.0) * torch.log(x)
            - beta * x
        )
        # log mật độ Normal
        log_normal = -0.5 * _LOG_2PI - torch.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2

        # log của hỗn hợp qua logsumexp -> ổn định số học
        log_mix = torch.logsumexp(
            torch.stack(
                [torch.log(pi) + log_gamma, torch.log1p(-pi) + log_normal], dim=0
            ),
            dim=0,
        )  # [B, G]

        nll = -log_mix
        if mask is not None:
            mask = mask.float()
            return (nll * mask).sum() / mask.sum().clamp_min(1.0)
        return nll.mean()


def gamma_normal_expectation(params: dict) -> torch.Tensor:
    """Giá trị kỳ vọng dưới hỗn hợp -> ma trận đã bù khuyết x̂.

    E[x] = pi * E[Gamma] + (1 - pi) * mu,  với E[Gamma(alpha,beta)] = alpha/beta.
    """
    pi = params["pi"]
    return pi * (params["alpha"] / params["beta"]) + (1.0 - pi) * params["mu"]


def dropout_probability(params: dict) -> torch.Tensor:
    """Trả về ma trận xác suất dropout pi cho từng (tế bào, gene)."""
    return params["pi"]