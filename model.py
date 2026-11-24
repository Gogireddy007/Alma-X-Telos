"""Alma: causal patch decoder for zero-shot probabilistic forecasting.

Design (vs. Telos-Small):
  * Direct quantile objective at EVERY token. Each patch token predicts the next `horizon` steps x 9 quantiles,
    so one sequence of N tokens gives N supervised forecast origins (Telos supervises one block per origin and
    spends capacity on a JEPA latent target the metric never sees).
  * Per-token causal frames: token i is normalised with the running mean/std of everything up to the end of its
    patch, then `asinh`-compressed. Targets of token i live in the same frame, so the loss is scale-free and
    the model never sees the future through normalisation.
  * patch 16 (Telos: 32) for finer resolution on short series / short horizons; 512-step direct horizon.
  * Plain, fast pre-norm transformer: RoPE, QK-norm, SwiGLU, causal SDPA.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class AlmaConfig:
    patch: int = 16
    horizon: int = 512
    d_model: int = 384
    n_layers: int = 10
    n_heads: int = 6
    ff_dim: int = 1024
    head_hidden: int = 1024
    quantiles: tuple = QUANTILES
    min_scale: float = 1e-3
    asinh_clip: float = 8.0
    rope_theta: float = 10000.0

    @property
    def n_q(self) -> int:
        return len(self.quantiles)


# ------------------------------------------------------------------------------------------------
# Causal frames
# ------------------------------------------------------------------------------------------------


def causal_frames(x: Tensor, m: Tensor, patch: int, min_scale: float) -> tuple[Tensor, Tensor]:
    """x, m: (B, T) with T % patch == 0. Returns per-token (B, N) mean / std of observed values in [0, end_i]."""
    mf = m.to(x.dtype)
    xm = x * mf
    n = mf.cumsum(-1)[:, patch - 1 :: patch]
    s1 = xm.cumsum(-1)[:, patch - 1 :: patch]
    s2 = (xm * x).cumsum(-1)[:, patch - 1 :: patch]
    nn_ = n.clamp_min(1.0)
    mu = s1 / nn_
    var = ((s2 - nn_ * mu * mu) / (nn_ - 1.0).clamp_min(1.0)).clamp_min(0.0)
    sd = var.sqrt().clamp_min(min_scale)
    return mu, sd


def to_frame(v: Tensor, mu: Tensor, sd: Tensor, clip: float) -> Tensor:
    return torch.asinh((v - mu) / sd).clamp(-clip, clip)


def from_frame(z: Tensor, mu: Tensor, sd: Tensor, clip: float) -> Tensor:
    return torch.sinh(z.clamp(-clip, clip)) * sd + mu


# ------------------------------------------------------------------------------------------------
# Layers
# ------------------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class ResMLP(nn.Module):
    def __init__(self, i: int, h: int, o: int) -> None:
        super().__init__()
        self.fc1, self.fc2, self.skip = nn.Linear(i, h), nn.Linear(h, o), nn.Linear(i, o)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.silu(self.fc1(x))) + self.skip(x)


def rope_cache(n: int, dim: int, theta: float, device, dtype) -> tuple[Tensor, Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    f = torch.outer(torch.arange(n, device=device, dtype=torch.float32), inv)
    return f.cos().to(dtype), f.sin().to(dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, c: AlmaConfig) -> None:
        super().__init__()
        self.h, self.hd = c.n_heads, c.d_model // c.n_heads
        self.n1, self.n2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.o = nn.Linear(c.d_model, c.d_model, bias=False)
        self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)
        self.w12 = nn.Linear(c.d_model, 2 * c.ff_dim, bias=False)
        self.w3 = nn.Linear(c.ff_dim, c.d_model, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, attn_mask: Tensor | None) -> Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.h, self.hd).unbind(2)
        q, k = apply_rope(self.qn(q), cos, sin), apply_rope(self.kn(k), cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if attn_mask is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.o(a.transpose(1, 2).reshape(b, n, d))
        g, u = self.w12(self.n2(x)).chunk(2, -1)
        return x + self.w3(F.silu(g) * u)


# ------------------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------------------


class Alma(nn.Module):
    def __init__(self, c: AlmaConfig) -> None:
        super().__init__()
        self.c = c
        self.embed = ResMLP(2 * c.patch + 3, 2 * c.d_model, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.d_model)
        self.head = ResMLP(c.d_model, c.head_hidden, c.horizon * c.n_q)
        self.apply(self._init)
        for blk in self.blocks:  # depth-scaled residual init
            nn.init.normal_(blk.o.weight, std=0.02 / math.sqrt(2 * c.n_layers))
            nn.init.normal_(blk.w3.weight, std=0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _init(mod: nn.Module) -> None:
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, std=0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

    def encode(self, x: Tensor, m: Tensor, extra: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """x, m: (B, T), T % patch == 0, x pre-scaled & zero where unobserved. Returns h (B,N,D), mu, sd (B,N)."""
        c = self.c
        b, t = x.shape
        n = t // c.patch
        mu, sd = causal_frames(x, m, c.patch, c.min_scale)
        xp = x.view(b, n, c.patch)
        mp = m.view(b, n, c.patch).to(x.dtype)
        z = to_frame(xp, mu[..., None], sd[..., None], c.asinh_clip) * mp
        mu_prev = torch.cat([mu[:, :1], mu[:, :-1]], 1)
        sd_prev = torch.cat([sd[:, :1], sd[:, :-1]], 1)
        feats = torch.stack(
            [torch.asinh((mu - mu_prev) / sd), torch.log(sd / sd_prev).clamp(-5, 5), mp.mean(-1)], -1
        )
        h = self.embed(torch.cat([z, mp, feats], -1))
        if extra is not None:  # e.g. action embedding (Alma-WM)
            h = h + extra
        # tokens before the first observed patch carry no information: let them attend only to themselves
        started = (m.view(b, n, c.patch).any(-1)).cummax(-1).values
        attn_mask = None
        if not bool(started.all()):
            causal = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
            attn_mask = (causal[None] & started[:, None, :]) | torch.eye(n, dtype=torch.bool, device=x.device)[None]
            attn_mask = attn_mask[:, None]
        cos, sin = rope_cache(n, c.d_model // c.n_heads // 2 * 2, c.rope_theta, x.device, h.dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        for blk in self.blocks:
            h = blk(h, cos, sin, attn_mask)
        return self.norm(h), mu, sd

    def forward(self, x: Tensor, m: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns quantiles in each token's frame (B, N, horizon, n_q), plus mu, sd (B, N)."""
        h, mu, sd = self.encode(x, m)
        b, n, _ = h.shape
        q = self.head(h).view(b, n, self.c.horizon, self.c.n_q)
        return q, mu, sd

    def last_token_quantiles(self, x: Tensor, m: Tensor) -> Tensor:
        """Inference: quantiles in data units (B, horizon, n_q) from the last token only."""
        h, mu, sd = self.encode(x, m)
        q = self.head(h[:, -1]).view(-1, self.c.horizon, self.c.n_q).float()
        return from_frame(q, mu[:, -1, None, None].float(), sd[:, -1, None, None].float(), self.c.asinh_clip)


def quantile_loss(
    q: Tensor, mu: Tensor, sd: Tensor, y: Tensor, ym: Tensor, quantiles: tuple, clip: float, norm: str = "token"
) -> Tensor:
    """q: (B,N,H,Q) frame-space predictions; y, ym: (B,N,H) raw targets & validity; mu, sd: (B,N)."""
    yz = to_frame(y, mu[..., None], sd[..., None], clip)
    qs = torch.tensor(quantiles, device=q.device, dtype=q.dtype)
    diff = yz[..., None] - q
    pin = torch.maximum(qs * diff, (qs - 1.0) * diff).mean(-1)
    w = ym.to(pin.dtype)
    tok = (pin * w).sum() / w.sum().clamp_min(1.0)
    if norm == "token":
        return tok
    has = w.sum((1, 2)) > 0  # per-series mean, then mean over series: every series counts equally
    ser = ((pin * w).sum((1, 2)) / w.sum((1, 2)).clamp_min(1.0))[has].mean()
    return ser if norm == "series" else 0.5 * (tok + ser)


def config_to_dict(c: AlmaConfig) -> dict:
    d = asdict(c)
    d["quantiles"] = list(c.quantiles)
    return d


def config_from_dict(d: dict) -> AlmaConfig:
    d = dict(d)
    d["quantiles"] = tuple(d["quantiles"])
    return AlmaConfig(**d)
