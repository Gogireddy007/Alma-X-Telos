"""Alma-WM2: JEPA world model with a frozen forecasting path and a linear-dynamics prior.

Changes vs Alma-WM (model_wm.py), each aimed at a failed wm1 gate:
  * fc_base  - a FROZEN copy of Alma v2. Action-free forecasts come from it, so real-data skill equals v2's
               by construction (gate 4.2 non-inferiority). The world-model encoder trains separately.
  * ARX prior - inside the model, a per-origin ridge ARX(p, q) is fitted on the context (values + actions,
               causal, closed form) and rolled forward under the planned actions. The decoder sees the prior
               and outputs  q = alpha * prior + residual, so the model contains the linear law it is
               compared against (gates 4, 5.1, 6.5*, 12). alpha is learned per step from the latent state.
  * JEPA core unchanged: context encoder -> latent predictor over action tokens -> EMA target encoder
               (stop-grad), L1 latent loss + variance anti-collapse.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model import Alma, AlmaConfig, ResMLP, config_from_dict, from_frame, quantile_loss, to_frame
from model_wm import N_ACT, AlmaA, Predictor, act_features


@dataclass
class WM2Config:
    pred_layers: int = 4
    K: int = 8
    chain: int = 2
    dec_hidden: int = 768
    n_origins: int = 8
    act_clip: float = 6.0
    arx_p: int = 24  # value lags
    arx_q: int = 24  # action lags per channel
    arx_fit: int = 768  # context steps used for the fit
    arx_ridge: float = 1e-2
    gate_bias: float = -6.0  # alpha ~ 0 at a warm start: starts as wm1


# ------------------------------------------------------------------------------------------------
# Differentiable-free (closed form, no grad) ARX prior
# ------------------------------------------------------------------------------------------------


@torch.no_grad()
def arx_prior(v: Tensor, vm: Tensor, u: Tensor, um: Tensor, L: int, w: WM2Config) -> Tensor:
    """Fit y_t = sum a_i y_{t-i} + sum b_ij u_{t-j} + c on the context window and roll forward L steps.

    v, vm: (M, W) normalised values and mask, the last column is the last observed step (the origin).
    u, um: (M, W + L, A) normalised actions over the same window plus the planned future.
    Returns (M, L) prior predictions in the same normalised units. Fitted in float32 on CPU-safe ops.
    """
    dt = v.dtype
    v, u = v.float(), u.float() * um.float()
    vm = vm.float()
    M, W = v.shape
    p, q = w.arx_p, w.arx_q
    A = u.shape[-1]
    r = max(p, q)
    # lag features for targets t = r .. W-1
    idx = torch.arange(r, W, device=v.device)
    ylag = torch.stack([v[:, idx - i] for i in range(1, p + 1)], -1)  # (M, n, p)
    ylm = torch.stack([vm[:, idx - i] for i in range(1, p + 1)], -1).amin(-1)
    ulag = torch.cat([u[:, idx - j] for j in range(1, q + 1)], -1)  # (M, n, q*A)
    X = torch.cat([ylag, ulag, torch.ones_like(ylag[..., :1])], -1)
    yt = v[:, idx]
    wt = (ylm * vm[:, idx])[..., None]
    Xw = X * wt
    G = Xw.transpose(1, 2) @ X
    nf = G.shape[-1]
    lam = w.arx_ridge * (G.diagonal(dim1=1, dim2=2).mean(-1, keepdim=True)[..., None] + 1e-3)
    G = G + lam * torch.eye(nf, device=v.device)
    b = Xw.transpose(1, 2) @ yt[..., None]
    theta = torch.linalg.solve(G.cpu(), b.cpu())[..., 0]  # (M, nf); MPS lacks batched solve
    ok = (wt.sum((1, 2)) > 3 * nf).float()[:, None]  # too little data -> prior is persistence
    dev = v.device
    theta, ok = theta.cpu(), ok.cpu()
    a, bu, c = theta[:, :p], theta[:, p : p + q * A], theta[:, -1:]
    # stability guard: shrink the AR part if the companion spectrum is outside the unit circle
    comp = torch.zeros(M, p, p)
    comp[:, 0] = a
    if p > 1:
        comp[:, 1:, :-1] = torch.eye(p - 1)
    rad = torch.linalg.eigvals(comp).abs().amax(-1)
    shrink = torch.where(rad > 0.999, 0.999 / rad, torch.ones_like(rad))
    a = a * shrink[:, None] ** torch.arange(1, p + 1)[None]
    # action contribution is not recursive: compute it for all future steps at once
    uc = u.cpu()
    tt = W + torch.arange(L)
    ufut = torch.cat([uc[:, tt - j] for j in range(1, q + 1)], -1)  # (M, L, q*A)
    drive = (ufut * bu[:, None]).sum(-1) + c  # (M, L)
    hist = torch.where(vm.bool(), v, torch.zeros_like(v)).cpu()
    last = hist[:, -1:]
    Y = torch.cat([hist[:, -p:], torch.zeros(M, L)], 1)
    a_rev = a.flip(-1)  # aligns with Y[:, k : k + p] (oldest .. newest)
    for k in range(L):
        Y[:, p + k] = ((Y[:, k : k + p] * a_rev).sum(-1) + drive[:, k]).clamp(-1e3, 1e3)
    pr = Y[:, p:]
    pr = ok * pr + (1 - ok) * last
    pr = pr.to(dev)
    return torch.nan_to_num(pr, 0.0).to(dt)


# ------------------------------------------------------------------------------------------------


class AlmaWM2(nn.Module):
    def __init__(self, c: AlmaConfig, w: WM2Config | None = None) -> None:
        super().__init__()
        self.c, self.w = c, w or WM2Config()
        self.enc = AlmaA(c)
        self.pred = Predictor(c, self.w)
        P = c.patch
        self.dec = ResMLP(2 * c.d_model + P, self.w.dec_hidden, P * c.n_q + P)  # residual quantiles + gates
        nn.init.zeros_(self.dec.fc2.weight[P * c.n_q :])
        nn.init.constant_(self.dec.fc2.bias[P * c.n_q :], self.w.gate_bias)
        nn.init.zeros_(self.dec.skip.weight[P * c.n_q :])
        nn.init.zeros_(self.dec.skip.bias[P * c.n_q :])
        self.fc_base = Alma(c)  # frozen Alma v2 forecasting path
        for p_ in self.fc_base.parameters():
            p_.requires_grad_(False)

    def trainable(self):
        return [p for n, p in self.named_parameters() if not n.startswith("fc_base.") and p.requires_grad]

    # -------------------------------------------------------------------------------------------
    def rollout(self, s0: Tensor, af: Tensor) -> Tensor:
        K, outs, s = self.w.K, [], s0
        for j in range(0, af.shape[1], K):
            z = self.pred(s, af[:, j : j + K])
            outs.append(z)
            s = z[:, -1]
        return torch.cat(outs, 1)

    def decode(self, z: Tensor, h0: Tensor, prior: Tensor) -> Tensor:
        """z: (M, L, D); h0: (M, D); prior: (M, L, patch) in the origin frame -> (M, L, patch, n_q)."""
        c, P = self.c, self.c.patch
        hh = h0[:, None].expand(-1, z.shape[1], -1)
        o = self.dec(torch.cat([z, hh, prior], -1))
        r = o[..., : P * c.n_q].view(*z.shape[:2], P, c.n_q)
        alpha = torch.sigmoid(o[..., P * c.n_q :])[..., None]
        return (alpha * prior[..., None] + r).sort(-1).values

    def prior_frame(self, x, m, u, um, mu0, sd0, n_steps):
        """ARX prior for the steps after the context end, mapped into the origin frame: (B, n_steps)."""
        T, Wf = x.shape[1], self.w.arx_fit
        s = max(0, T - Wf)
        v = (x[:, s:] - mu0[:, None]) / sd0[:, None]
        pr = arx_prior(v, m[:, s:], u[:, s : T + n_steps], um[:, s : T + n_steps], n_steps, self.w)
        return to_frame(pr, 0.0, 1.0, self.c.asinh_clip)

    @torch.no_grad()
    def forecast(self, x: Tensor, m: Tensor, u: Tensor, um: Tensor, n_future: int) -> Tensor:
        T, P = x.shape[1], self.c.patch
        h, mu, sd = self.enc.encode(x, m, u[:, :T], um[:, :T])
        af = act_features(u[:, T:], um[:, T:].to(x.dtype), P, self.w.act_clip)
        z = self.rollout(h[:, -1], af[:, :n_future])
        mu0, sd0 = mu[:, -1].reshape(-1), sd[:, -1].reshape(-1)
        pr = self.prior_frame(x, m, u, um, mu0, sd0, n_future * P).view(-1, n_future, P)
        q = self.decode(z, h[:, -1], pr).flatten(1, 2)
        return from_frame(q.float(), mu0[:, None, None].float(), sd0[:, None, None].float(), self.c.asinh_clip)

    @torch.no_grad()
    def decode_states(self, z, x, m, u, um):
        """Decode given future latents z (B, L, D) from context x (closed-loop gate 5.2)."""
        T, P = x.shape[1], self.c.patch
        h, mu, sd = self.enc.encode(x, m, u[:, :T], um[:, :T])
        mu0, sd0 = mu[:, -1].reshape(-1), sd[:, -1].reshape(-1)
        pr = self.prior_frame(x, m, u, um, mu0, sd0, z.shape[1] * P).view(-1, z.shape[1], P)
        q = self.decode(z, h[:, -1], pr).flatten(1, 2)
        return from_frame(q.float(), mu0[:, None, None].float(), sd0[:, None, None].float(), self.c.asinh_clip)

    @torch.no_grad()
    def state_rollout(self, x, m, u, um, n_future):
        T = x.shape[1]
        h, _, _ = self.enc.encode(x, m, u[:, :T], um[:, :T])
        af = act_features(u[:, T:], um[:, T:].to(x.dtype), self.c.patch, self.w.act_clip)
        return h[:, -1], self.rollout(h[:, -1], af[:, :n_future])


def wm2_losses(model: AlmaWM2, target: AlmaA, x, m, u, um, y, ym, gen=None) -> dict:
    c, w = model.c, model.w
    P, B, T = c.patch, x.shape[0], x.shape[1]
    N = T // P
    h, mu, sd = model.enc.encode(x, m, u, um)
    q = model.enc.head(h).view(B, N, c.horizon, c.n_q)
    l_direct = quantile_loss(q, mu, sd, y, ym, c.quantiles, c.asinh_clip)  # representation shaping only

    with torch.no_grad():
        zt, _, _ = target.encode(x, m, u, um)
    af = act_features(u, um.to(x.dtype), P, w.act_clip)

    L, R = w.K * w.chain, w.n_origins
    i0 = torch.randint(4, N - L - 1, (B, R), device=x.device, generator=gen)
    bi = torch.arange(B, device=x.device)[:, None].expand(B, R)
    steps = i0[..., None] + 1 + torch.arange(L, device=x.device)
    s0 = h[bi, i0].reshape(B * R, -1)
    a_f = af[bi[..., None], steps].reshape(B * R, L, -1)
    z_tgt = zt[bi[..., None], steps].reshape(B * R, L, -1)
    xp, mp = x.view(B, N, P), m.view(B, N, P)
    yv = xp[bi[..., None], steps].reshape(B * R, L, P)
    yvm = mp[bi[..., None], steps].reshape(B * R, L, P)
    mu0, sd0 = mu[bi, i0].reshape(-1), sd[bi, i0].reshape(-1)
    started = mp.any(-1).to(torch.int8).cummax(-1).values.bool()[bi, i0].reshape(-1)
    valid = (yvm.any(-1) & started[:, None]).to(x.dtype)

    # ARX prior per origin: fit window ends at the end of the origin patch
    Wf, LS = w.arx_fit, L * P
    pad = Wf
    xpd = torch.cat([x.new_zeros(B, pad), x], 1)
    mpd = torch.cat([m.new_zeros(B, pad), m], 1)
    upd = torch.cat([u.new_zeros(B, pad, u.shape[-1]), u], 1)
    umd = torch.cat([um.new_zeros(B, pad, um.shape[-1]), um], 1)
    end = (i0 + 1) * P + pad  # exclusive end of the context in padded coordinates
    off = torch.arange(-Wf, 0, device=x.device)
    offu = torch.arange(-Wf, LS, device=x.device)
    ix = (end[..., None] + off).reshape(B * R, Wf)
    ixu = (end[..., None] + offu).reshape(B * R, Wf + LS)
    br = bi.reshape(-1)[:, None]
    vwin = (xpd[br, ix] - mu0[:, None]) / sd0[:, None]
    pr = arx_prior(vwin, mpd[br, ix], upd[br, ixu], umd[br, ixu], LS, w)
    prf = to_frame(pr, 0.0, 1.0, c.asinh_clip).view(B * R, L, P)

    z_hat = model.rollout(s0, a_f)
    ln = lambda t: F.layer_norm(t, t.shape[-1:])
    l_jepa = ((ln(z_hat) - ln(z_tgt)).abs().mean(-1) * valid).sum() / valid.sum().clamp_min(1)
    std = ln(z_hat).reshape(-1, z_hat.shape[-1]).float().std(0)
    l_var = F.relu(1.0 - std / (ln(z_tgt).reshape(-1, z_tgt.shape[-1]).float().std(0) + 1e-4)).mean()

    qs = torch.tensor(c.quantiles, device=x.device, dtype=x.dtype)
    yz = to_frame(yv, mu0[:, None, None], sd0[:, None, None], c.asinh_clip)
    wgt = yvm.to(x.dtype) * started[:, None, None].to(x.dtype)

    def pin(qv: Tensor) -> Tensor:
        d = yz[..., None] - qv
        p = torch.maximum(qs * d, (qs - 1) * d).mean(-1)
        return (p * wgt).sum() / wgt.sum().clamp_min(1)

    l_ground = pin(model.decode(z_hat, s0, prf))
    l_teacher = pin(model.decode(z_tgt, s0, prf))
    l_prior = ((yz - prf).abs() * wgt).sum() / wgt.sum().clamp_min(1)  # monitoring only (no grad)
    total = l_direct + l_ground + 0.5 * l_teacher + l_jepa + 1.0 * l_var
    return {"total": total, "direct": l_direct, "ground": l_ground, "teacher": l_teacher, "jepa": l_jepa,
            "var": l_var, "z_std": std.mean(), "prior_mae": l_prior.detach()}


def wm2_config_dict(c: AlmaConfig, w: WM2Config) -> dict:
    d = asdict(c)
    d["quantiles"] = list(c.quantiles)
    return {"base": d, "wm": asdict(w), "version": 2}


def load_wm2(path: str, device: str = "cpu", which: str = "ema") -> AlmaWM2:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = AlmaWM2(config_from_dict(cfg["base"]), WM2Config(**cfg["wm"]))
    sd = dict(ck[which] if which in ck else ck["ema"])
    if not any(k.startswith("fc_base.") for k in sd):  # frozen v2 path is stored once, in the base checkpoint
        b = torch.load(ck["base_from"], map_location="cpu", weights_only=False)
        sd.update({"fc_base." + k: v for k, v in b.get("ema", b.get("model")).items()})
    model.load_state_dict(sd)
    return model.to(device).eval()


def load_any(path: str, device: str = "cpu", which: str = "ema"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["config"].get("version") == 2:
        return load_wm2(path, device, which)
    from model_wm import load_wm

    return load_wm(path, device, which)
