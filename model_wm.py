"""Alma-WM: an action-conditioned JEPA world model for time series.

  history x<=t, actions a<=t --> context encoder E_theta (Alma backbone + action embedding) --> state s_t
  s_t, planned actions a(t:t+K] --> latent predictor P_psi --> z^(t+1..t+K)        (dynamics in latent space)
  x<=t+k, a<=t+k --> target encoder E_xi (EMA of E_theta, stop-grad) --> z(t+k)    (JEPA target)
  z^(t+k), s_t --> decoder D --> quantiles of patch t+k                            (grounding, values)

Losses: JEPA (L1 between layer-normed predicted and target latents, over chained rollouts),
variance floor (anti-collapse), grounding pinball loss on decoded rollouts (open-loop and teacher),
and Alma's direct quantile loss (action-free forecast head, keeps the GIFT-Eval skill).
Long rollouts chain the predictor: the last predicted latent of a chunk becomes the next state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model import Alma, AlmaConfig, Block, ResMLP, RMSNorm, config_from_dict, rope_cache, to_frame, from_frame

N_ACT = 2


@dataclass
class WMConfig:
    pred_layers: int = 4
    K: int = 8  # patches per predictor chunk (8 x 16 = 128 steps)
    chain: int = 2  # chunks chained during training (open-loop rollout of 256 steps)
    dec_hidden: int = 768
    n_origins: int = 8  # forecast origins per sequence for the world-model losses
    act_clip: float = 6.0


def act_features(u: Tensor, um: Tensor, patch: int, clip: float) -> Tensor:
    """u, um: (B, T, A) pre-scaled actions & presence. Returns per-patch features (B, N, A*patch + A)."""
    b, t, a = u.shape
    n = t // patch
    v = torch.asinh(u.clamp(-1e4, 1e4)).clamp(-clip, clip) * um
    v = v.view(b, n, patch, a).transpose(2, 3).reshape(b, n, a * patch)
    pres = um.view(b, n, patch, a).mean(2)
    return torch.cat([v, pres], -1)


class AlmaA(Alma):
    """Alma backbone whose tokens also see the actions applied during their patch."""

    def __init__(self, c: AlmaConfig) -> None:
        super().__init__(c)
        self.act_embed = ResMLP(N_ACT * c.patch + N_ACT, c.d_model, c.d_model)
        for lin in (self.act_embed.fc2, self.act_embed.skip):  # starts as a no-op on a warm start
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def encode(self, x: Tensor, m: Tensor, u: Tensor | None = None, um: Tensor | None = None):
        if u is None:
            u = x.new_zeros(*x.shape, N_ACT)
            um = u.clone()
        extra = self.act_embed(act_features(u, um.to(x.dtype), self.c.patch, 6.0))
        return super().encode(x, m, extra)


class Predictor(nn.Module):
    """Causal transformer: [state] + K action tokens -> K next latent states."""

    def __init__(self, c: AlmaConfig, w: WMConfig) -> None:
        super().__init__()
        self.c, self.w = c, w
        d = c.d_model
        self.s_in = nn.Linear(d, d)
        self.a_in = ResMLP(N_ACT * c.patch + N_ACT, d, d)
        self.step = nn.Parameter(torch.randn(w.K, d) * 0.02)
        self.blocks = nn.ModuleList([Block(c) for _ in range(w.pred_layers)])
        self.norm = RMSNorm(d)
        self.out = nn.Linear(d, d)

    def forward(self, s: Tensor, af: Tensor) -> Tensor:
        """s: (M, D) state; af: (M, K, A*patch+A) action features of the next K patches. Returns (M, K, D)."""
        tok = torch.cat([self.s_in(s)[:, None], self.a_in(af) + self.step[None, : af.shape[1]]], 1)
        n = tok.shape[1]
        cos, sin = rope_cache(n, self.c.d_model // self.c.n_heads // 2 * 2, self.c.rope_theta, s.device, tok.dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        for blk in self.blocks:
            tok = blk(tok, cos, sin, None)
        return self.out(self.norm(tok[:, 1:]))


class AlmaWM(nn.Module):
    def __init__(self, c: AlmaConfig, w: WMConfig | None = None) -> None:
        super().__init__()
        self.c, self.w = c, w or WMConfig()
        self.enc = AlmaA(c)
        self.pred = Predictor(c, self.w)
        self.dec = ResMLP(2 * c.d_model, self.w.dec_hidden, c.patch * c.n_q)

    # -------------------------------------------------------------------------------------------
    def rollout(self, s0: Tensor, af: Tensor) -> Tensor:
        """Chain the predictor over all patches in af (M, L, F). Returns latents (M, L, D)."""
        K, outs, s = self.w.K, [], s0
        for j in range(0, af.shape[1], K):
            z = self.pred(s, af[:, j : j + K])
            outs.append(z)
            s = z[:, -1]
        return torch.cat(outs, 1)

    def decode(self, z: Tensor, h0: Tensor) -> Tensor:
        """z: (M, L, D) latents, h0: (M, D) origin state -> (M, L, patch, n_q) quantiles in the origin frame."""
        hh = h0[:, None].expand(-1, z.shape[1], -1)
        q = self.dec(torch.cat([z, hh], -1))
        q = q.view(*z.shape[:2], self.c.patch, self.c.n_q)
        return q.sort(-1).values

    @torch.no_grad()
    def forecast(self, x: Tensor, m: Tensor, u: Tensor, um: Tensor, n_future: int) -> Tensor:
        """x, m: (B, T) context (T % patch == 0, pre-scaled); u, um: (B, T + n_future*patch, A) actions over
        context AND the planned future. Returns quantiles (B, n_future*patch, n_q) in the pre-scaled units."""
        T = x.shape[1]
        P = self.c.patch
        h, mu, sd = self.enc.encode(x, m, u[:, :T], um[:, :T])
        af = act_features(u[:, T:], um[:, T:].to(x.dtype), P, self.w.act_clip)
        z = self.rollout(h[:, -1], af[:, :n_future])
        q = self.decode(z, h[:, -1]).flatten(1, 2)
        return from_frame(q.float(), mu[:, -1, None, None].float(), sd[:, -1, None, None].float(), self.c.asinh_clip)

    @torch.no_grad()
    def state_rollout(self, x, m, u, um, n_future):
        """Latent trajectory (B, n_future, D) - used by the validation gates (probes, drift, OOD)."""
        T = x.shape[1]
        h, _, _ = self.enc.encode(x, m, u[:, :T], um[:, :T])
        af = act_features(u[:, T:], um[:, T:].to(x.dtype), self.c.patch, self.w.act_clip)
        return h[:, -1], self.rollout(h[:, -1], af[:, :n_future])


def wm_losses(model: AlmaWM, target: AlmaA, x, m, u, um, y, ym, gen: torch.Generator | None = None) -> dict:
    """All training losses for one batch. x, m: (B,T); u, um: (B,T,A); y, ym: direct-head targets (B,N,H)."""
    from model import quantile_loss

    c, w = model.c, model.w
    P, B, T = c.patch, x.shape[0], x.shape[1]
    N = T // P
    h, mu, sd = model.enc.encode(x, m, u, um)
    q = model.enc.head(h).view(B, N, c.horizon, c.n_q)
    l_direct = quantile_loss(q, mu, sd, y, ym, c.quantiles, c.asinh_clip)

    with torch.no_grad():
        zt, _, _ = target.encode(x, m, u, um)
    af = act_features(u, um.to(x.dtype), P, w.act_clip)  # (B, N, F)

    L = w.K * w.chain
    R = w.n_origins
    lo = 4
    hi = N - L - 1
    i0 = torch.randint(lo, hi, (B, R), device=x.device, generator=gen)  # origin token per sample
    bi = torch.arange(B, device=x.device)[:, None].expand(B, R)
    steps = i0[..., None] + 1 + torch.arange(L, device=x.device)  # (B,R,L) future token indices
    s0 = h[bi, i0].reshape(B * R, -1)
    a_f = af[bi[..., None], steps].reshape(B * R, L, -1)
    z_tgt = zt[bi[..., None], steps].reshape(B * R, L, -1)
    xp, mp = x.view(B, N, P), m.view(B, N, P)
    yv = xp[bi[..., None], steps].reshape(B * R, L, P)
    yvm = mp[bi[..., None], steps].reshape(B * R, L, P)
    mu0, sd0 = mu[bi, i0].reshape(-1), sd[bi, i0].reshape(-1)
    tok_started = mp.any(-1).to(torch.int8).cummax(-1).values.bool()  # (B, N): any data observed so far
    started = tok_started[bi, i0].reshape(-1)
    valid = (yvm.any(-1) & started[:, None]).to(x.dtype)  # (M, L)

    z_hat = model.rollout(s0, a_f)  # open-loop chained rollout in latent space
    ln = lambda t: F.layer_norm(t, t.shape[-1:])
    l_jepa = ((ln(z_hat) - ln(z_tgt)).abs().mean(-1) * valid).sum() / valid.sum().clamp_min(1)
    std = ln(z_hat).reshape(-1, z_hat.shape[-1]).float().std(0)
    l_var = F.relu(1.0 - std / (ln(z_tgt).reshape(-1, z_tgt.shape[-1]).float().std(0) + 1e-4)).mean()

    qs = torch.tensor(c.quantiles, device=x.device, dtype=x.dtype)

    def pin(qv: Tensor) -> Tensor:
        yz = to_frame(yv, mu0[:, None, None], sd0[:, None, None], c.asinh_clip)
        d = yz[..., None] - qv
        p = torch.maximum(qs * d, (qs - 1) * d).mean(-1)
        wgt = yvm.to(x.dtype) * started[:, None, None].to(x.dtype)
        return (p * wgt).sum() / wgt.sum().clamp_min(1)

    l_ground = pin(model.decode(z_hat, s0))
    l_teacher = pin(model.decode(z_tgt, s0))
    total = l_direct + l_ground + 0.5 * l_teacher + l_jepa + 1.0 * l_var
    return {"total": total, "direct": l_direct, "ground": l_ground, "teacher": l_teacher, "jepa": l_jepa,
            "var": l_var, "z_std": std.mean()}


def wm_config_dict(c: AlmaConfig, w: WMConfig) -> dict:
    d = asdict(c)
    d["quantiles"] = list(c.quantiles)
    return {"base": d, "wm": asdict(w)}


def load_wm(path: str, device: str = "cpu", which: str = "ema") -> AlmaWM:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = AlmaWM(config_from_dict(cfg["base"]), WMConfig(**cfg["wm"]))
    model.load_state_dict(ck[which] if which in ck else ck["ema"])
    return model.to(device).eval()
