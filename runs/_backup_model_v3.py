"""Alma v3: 12 x 512 forecaster with a stride-32 coarse path and a JEPA latent objective.

  * Grown from Alma v2 (8 x 384) by an exact, function-preserving width + depth expansion (grow_from_v2).
  * Fine path: the last `fine_len` (4,096) steps as patch-16 tokens, exactly as in v2.
  * Coarse path: anything older (up to 8,192 total) as stride-32 tokens with their own embedding. All coarse
    tokens precede all fine tokens, so causality is kept. RoPE positions are the token end times in units of
    16 steps, so the coarse and fine tokens share one time axis. Output heads read only the fine tokens.
  * JEPA: from token i, a horizon-conditioned latent predictor (the same Predictor module the world model
    uses, with no actions) predicts the EMA target encoder's latents at tokens i+1..i+K. Forecaster and
    world model therefore share one JEPA backbone.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model import Alma, AlmaConfig, ResMLP, apply_rope, causal_frames, to_frame


@dataclass
class V3Config(AlmaConfig):
    patch: int = 16
    horizon: int = 1024
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    ff_dim: int = 1536
    head_hidden: int = 1536
    coarse_patch: int = 32
    fine_len: int = 4096
    jepa_K: int = 8
    jepa_layers: int = 2


def v3_config_dict(c: V3Config) -> dict:
    d = asdict(c)
    d["quantiles"] = list(c.quantiles)
    d["arch"] = "v3"
    return d


def v3_config_from_dict(d: dict) -> V3Config:
    d = {k: v for k, v in d.items() if k != "arch"}
    d["quantiles"] = tuple(d["quantiles"])
    return V3Config(**d)


def rope_at(pos: Tensor, dim: int, theta: float, dtype) -> tuple[Tensor, Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device, dtype=torch.float32) / dim))
    f = pos.float()[:, None] * inv[None]
    return f.cos().to(dtype), f.sin().to(dtype)


class AlmaV3(Alma):
    def __init__(self, c: V3Config) -> None:
        super().__init__(c)
        cp = c.coarse_patch
        self.embed_c = ResMLP(2 * cp + 3, 2 * c.d_model, c.d_model)
        self.type_c = nn.Parameter(torch.zeros(c.d_model))

    def _embed(self, x, m, P, emb):
        b, t = x.shape
        n = t // P
        mu, sd = causal_frames(x, m, P, self.c.min_scale)
        xp = x.view(b, n, P)
        mp = m.view(b, n, P).to(x.dtype)
        z = to_frame(xp, mu[..., None], sd[..., None], self.c.asinh_clip) * mp
        mu_prev = torch.cat([mu[:, :1], mu[:, :-1]], 1)
        sd_prev = torch.cat([sd[:, :1], sd[:, :-1]], 1)
        feats = torch.stack([torch.asinh((mu - mu_prev) / sd), torch.log(sd / sd_prev).clamp(-5, 5), mp.mean(-1)], -1)
        return emb(torch.cat([z, mp, feats], -1)), mu, sd, m.view(b, n, P).any(-1)

    def encode(self, x: Tensor, m: Tensor, extra: Tensor | None = None):
        c = self.c
        P, cp, F_ = c.patch, c.coarse_patch, c.fine_len
        b, t = x.shape
        if t <= F_:
            return super().encode(x, m, extra)
        pre = t - F_
        if pre % cp:  # left-pad the coarse prefix to a whole number of coarse patches
            pad = cp - pre % cp
            x = torch.cat([x.new_zeros(b, pad), x], 1)
            m = torch.cat([m.new_zeros(b, pad), m], 1)
            pre += pad
        # frames: fine tokens use stats over EVERYTHING up to their end (same as a long v2 context)
        mu_all, sd_all = causal_frames(x, m, P, c.min_scale)
        nf = F_ // P
        mu, sd = mu_all[:, -nf:], sd_all[:, -nf:]
        xf, mf = x[:, pre:], m[:, pre:]
        xp = xf.view(b, nf, P)
        mp = mf.view(b, nf, P).to(x.dtype)
        z = to_frame(xp, mu[..., None], sd[..., None], c.asinh_clip) * mp
        mu_prev = mu_all[:, -nf - 1 : -1]
        sd_prev = sd_all[:, -nf - 1 : -1]
        feats = torch.stack([torch.asinh((mu - mu_prev) / sd), torch.log(sd / sd_prev).clamp(-5, 5), mp.mean(-1)], -1)
        hf = self.embed(torch.cat([z, mp, feats], -1))
        if extra is not None:
            hf = hf + extra
        hc, _, _, oc = self._embed(x[:, :pre], m[:, :pre], cp, self.embed_c)
        hc = hc + self.type_c
        nc = pre // cp
        h = torch.cat([hc, hf], 1)
        obs = torch.cat([oc, mf.view(b, nf, P).any(-1)], 1)
        n = nc + nf
        started = obs.cummax(-1).values
        attn_mask = None
        if not bool(started.all()):
            causal = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
            attn_mask = ((causal[None] & started[:, None, :]) | torch.eye(n, dtype=torch.bool, device=x.device)[None])[:, None]
        r = cp // P
        pos = torch.cat([torch.arange(nc, device=x.device) * r + (r - 1), pre // P + torch.arange(nf, device=x.device)])
        cos, sin = rope_at(pos, c.d_model // c.n_heads // 2 * 2, c.rope_theta, h.dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        for blk in self.blocks:
            h = blk(h, cos, sin, attn_mask)
        return self.norm(h)[:, nc:], mu, sd


# ------------------------------------------------------------------------------------------------
# Exact growth 8 x 384 (v2) -> 12 x 512 (v3)
# ------------------------------------------------------------------------------------------------


@torch.no_grad()
def grow_from_v2(model: AlmaV3, sd_old: dict, c_old: AlmaConfig) -> list[int]:
    """Copy v2 weights into the larger v3 so that v3(x) == v2(x) for contexts <= fine_len.

    New residual dimensions start at zero and every write into them is zero; RMSNorm gains are rescaled by
    sqrt(d_old / d_new) so normalised old dimensions are unchanged; new heads, FFN units and hidden units have
    zero output weights; inserted blocks have zero output projections (identity)."""
    c = model.c
    do, dn = c_old.d_model, c.d_model
    s = math.sqrt(do / dn)
    own = model.state_dict()
    hd = do // c_old.n_heads
    assert hd == dn // c.n_heads
    Lo, Ln = c_old.n_layers, c.n_layers
    idx = [int(round(j * Ln / Lo)) for j in range(Lo)]
    ffo, ffn = c_old.ff_dim, c.ff_dim

    def norm_w(new, old):
        new.fill_(s)
        new[:do] = old * s

    # embed (ResMLP i -> 2d -> d)
    e = "embed."
    own[e + "fc1.weight"][: 2 * do] = sd_old[e + "fc1.weight"]
    own[e + "fc1.bias"][: 2 * do] = sd_old[e + "fc1.bias"]
    own[e + "fc2.weight"].zero_()
    own[e + "fc2.weight"][:do, : 2 * do] = sd_old[e + "fc2.weight"]
    own[e + "fc2.bias"].zero_()
    own[e + "fc2.bias"][:do] = sd_old[e + "fc2.bias"]
    own[e + "skip.weight"].zero_()
    own[e + "skip.weight"][:do] = sd_old[e + "skip.weight"]
    own[e + "skip.bias"].zero_()
    own[e + "skip.bias"][:do] = sd_old[e + "skip.bias"]
    # Coarse embedding: start as the grown fine embedding applied to the pair-averaged stride-32 patch
    # (input slot j of a coarse patch maps to fine slot j // 2 with half weight). A zero-output init would
    # put every coarse token at RMSNorm's singular point (x = 0), which amplifies its gradient ~1/sqrt(eps)
    # per layer and overflowed the gradient norm on 8k batches.
    P_, cp_ = model.c.patch, model.c.coarse_patch
    r_ = cp_ // P_
    jmap = torch.arange(cp_) // r_
    for lin in ("fc1", "skip"):
        W = own[f"embed.{lin}.weight"]
        Wc = own[f"embed_c.{lin}.weight"]
        Wc.zero_()
        Wc[:, :cp_] = W[:, jmap] / r_
        Wc[:, cp_ : 2 * cp_] = W[:, P_ + jmap] / r_
        Wc[:, 2 * cp_ :] = W[:, 2 * P_ :]
        own[f"embed_c.{lin}.bias"].copy_(own[f"embed.{lin}.bias"])
    own["embed_c.fc2.weight"].copy_(own["embed.fc2.weight"])
    own["embed_c.fc2.bias"].copy_(own["embed.fc2.bias"])

    for i in range(Ln):
        b = f"blocks.{i}."
        if i not in idx:  # inserted identity block
            own[b + "o.weight"].zero_()
            own[b + "w3.weight"].zero_()
            own[b + "n1.w"].fill_(1.0)
            own[b + "n2.w"].fill_(1.0)
            continue
        j = idx.index(i)
        o = f"blocks.{j}."
        norm_w(own[b + "n1.w"], sd_old[o + "n1.w"])
        norm_w(own[b + "n2.w"], sd_old[o + "n2.w"])
        own[b + "qn.w"].copy_(sd_old[o + "qn.w"])
        own[b + "kn.w"].copy_(sd_old[o + "kn.w"])
        qn = own[b + "qkv.weight"].view(3, c.n_heads, hd, dn)
        qo = sd_old[o + "qkv.weight"].view(3, c_old.n_heads, hd, do)
        qn[:, : c_old.n_heads, :, :do] = qo
        qn[:, : c_old.n_heads, :, do:] = 0
        ow = own[b + "o.weight"]
        ow.zero_()
        ow[:do, : c_old.n_heads * hd] = sd_old[o + "o.weight"]
        w12 = own[b + "w12.weight"]
        w12o = sd_old[o + "w12.weight"]
        w12[:ffo, :do] = w12o[:ffo]
        w12[:ffo, do:] = 0
        w12[ffn : ffn + ffo, :do] = w12o[ffo:]
        w12[ffn : ffn + ffo, do:] = 0
        w3 = own[b + "w3.weight"]
        w3.zero_()
        w3[:do, :ffo] = sd_old[o + "w3.weight"]
    norm_w(own["norm.w"], sd_old["norm.w"])
    hh = c_old.head_hidden
    own["head.fc1.weight"][:hh, :do] = sd_old["head.fc1.weight"]
    own["head.fc1.weight"][:hh, do:] = 0
    own["head.fc1.bias"][:hh] = sd_old["head.fc1.bias"]
    own["head.fc2.weight"].zero_()
    own["head.fc2.weight"][:, :hh] = sd_old["head.fc2.weight"]
    own["head.fc2.bias"].copy_(sd_old["head.fc2.bias"])
    own["head.skip.weight"].zero_()
    own["head.skip.weight"][:, :do] = sd_old["head.skip.weight"]
    own["head.skip.bias"].copy_(sd_old["head.skip.bias"])
    model.load_state_dict(own)
    return idx


# ------------------------------------------------------------------------------------------------
# Losses
# ------------------------------------------------------------------------------------------------


def balanced_quantile_loss(q, mu, sd, y, ym, quantiles, clip, gamma: float = 0.5) -> Tensor:
    """Pinball loss averaged per horizon step first (each step weighted equally whatever its number of valid
    targets), then weighted by (1 + h/H)^gamma, so long horizons are not under-trained."""
    yz = to_frame(y, mu[..., None], sd[..., None], clip)
    qs = torch.tensor(quantiles, device=q.device, dtype=q.dtype)
    d = yz[..., None] - q
    pin = torch.maximum(qs * d, (qs - 1.0) * d).mean(-1)  # (B, N, H)
    w = ym.to(pin.dtype)
    per_h = (pin * w).sum((0, 1)) / w.sum((0, 1)).clamp_min(1.0)  # (H,)
    H = per_h.shape[0]
    hw = (1 + torch.arange(H, device=q.device, dtype=q.dtype) / H) ** gamma
    hw = hw * (w.sum((0, 1)) > 0)
    return (per_h * hw).sum() / hw.sum().clamp_min(1e-6)


class JEPAHead(nn.Module):
    """Horizon-conditioned latent predictor: same Predictor module as the world model, with no actions."""

    def __init__(self, c: V3Config) -> None:
        super().__init__()
        from model_wm import N_ACT, Predictor, WMConfig

        self.n_feat = N_ACT * c.patch + N_ACT
        self.pred = Predictor(c, WMConfig(pred_layers=c.jepa_layers, K=c.jepa_K))

    def forward(self, s: Tensor, K: int) -> Tensor:
        return self.pred(s, s.new_zeros(s.shape[0], K, self.n_feat))


def jepa_loss(head: JEPAHead, h: Tensor, zt: Tensor, valid_tok: Tensor, K: int, R: int = 16) -> tuple[Tensor, Tensor]:
    """h: (B,N,D) context latents; zt: (B,N,D) target-encoder latents (no grad); valid_tok: (B,N) bool."""
    B, N, D = h.shape
    i0 = torch.randint(0, N - K - 1, (B, R), device=h.device)
    bi = torch.arange(B, device=h.device)[:, None].expand(B, R)
    steps = i0[..., None] + 1 + torch.arange(K, device=h.device)
    s0 = h[bi, i0].reshape(B * R, D)
    tgt = zt[bi[..., None], steps].reshape(B * R, K, D)
    val = (valid_tok[bi[..., None], steps] & valid_tok[bi, i0][..., None]).reshape(B * R, K).to(h.dtype)
    zh = head(s0, K)
    ln = lambda t: F.layer_norm(t, t.shape[-1:])
    l = ((ln(zh) - ln(tgt)).abs().mean(-1) * val).sum() / val.sum().clamp_min(1)
    std = ln(zh).reshape(-1, D).float().std(0)
    lv = F.relu(1.0 - std / (ln(tgt).reshape(-1, D).float().std(0) + 1e-4)).mean()
    return l, lv


def load_v3(path: str, device: str = "cpu", which: str = "ema") -> AlmaV3:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = AlmaV3(v3_config_from_dict(ck["config"]))
    sd = ck[which] if which in ck else ck["ema"]
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("jepa.")})
    return m.to(device).eval()
